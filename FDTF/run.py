from data_helper import fdtf_config
from fdtf import FDTF
from utils import make_dir, strRed, strBlue
import json
import os
import numpy as np
import pickle
from multiprocessing import Lock, Pool

output_lock = Lock()


def run_fdtf_exp(data_str, course_str, model_str, fold, concept_dim, lambda_t,
                 lambda_q, lambda_bias, slr, lr, max_iter, metrics, log_file, validation, validation_limit = 30, test_range = None):
    """
    cross validation on the first fold of dataset to tune the hyper-parameters,
    then we will use those best hyper-parameters evaluate the performance on all folds
    """

    if course_str == "Quiz" or course_str == "QuizIceBreaker":
        views = "100"
    elif course_str == "Lecture":
        views = "010"
    elif course_str == "Discussion":
        views = "001"
    elif course_str == "Quiz_Lecture":
        views = "110"
    elif course_str == "Quiz_Discussion":
        views = "101"
    elif course_str == "Quiz_Lecture_Discussion":
        views = "111"
    else:
        raise IOError

    model_config = fdtf_config(
        data_str, course_str, views, concept_dim, fold, lambda_t, lambda_q,
        lambda_bias, slr, lr, max_iter, metrics, log_file, validation, validation_limit, test_range)


    if model_str == 'fdtf':
        model = FDTF(model_config)
    else:
        raise EnvironmentError("ERROR!!")

    if validation is True:
        test_data = model_config['val']
    else:
        test_data = model_config['test']
        model.train_data.extend(model_config['val'])

    # since the test start attempt for different students are different,
    # we need to find the first testing attempt, and add all lectures and discussion before
    # test_start_attempt into train_data

    test_start_attempt = None
    for (student, attempt, material, score, resource) in sorted(test_data, key=lambda x: x[1]):
        if resource == 0:
            test_start_attempt = attempt
            break
        else:
            model.train_data.append((student, attempt, material, score, resource))
    if test_start_attempt is None:
        raise EnvironmentError

    perf_dict = {}
    for test_attempt in range(test_start_attempt, model.num_attempts):
        model.current_test_attempt = test_attempt
        model.lr = lr
        restart_training(model)
        train_perf = model.training()

        test_set = []
        for (student, attempt, material, score, resource) in test_data:
            if attempt == model.current_test_attempt:
                test_set.append((student, attempt, material, score, resource))
                model.train_data.append((student, attempt, material, score, resource))

        test_perf = model.testing(test_set)
        if test_attempt not in perf_dict:
            perf_dict[test_attempt] = {}
            perf_dict[test_attempt]['train'] = train_perf
            if validation:
                perf_dict[test_attempt]['val'] = test_perf
            else:
                perf_dict[test_attempt]['test'] = test_perf

    overall_perf = model.eval(model.test_obs_list, model.test_pred_list)
    if validation:
        perf_dict['val'] = overall_perf
    else:
        perf_dict['test'] = overall_perf


    for stu in range(0, model.num_users):
        if stu in model_config['max_stu_attempt']:
            max_att = model_config['max_stu_attempt'][stu]
            if max_att < model.num_attempts - 1:
                model.T[stu,max_att + 2:] = model.T[stu, max_att + 1]

    save_exp_results(model, perf_dict, data_str, course_str, model_str, fold,
                     concept_dim, lambda_t, lambda_q,  lambda_bias, slr, lr,max_iter, validation=validation)


def restart_training(model):
    # initialize the bias for each attempt, student, question, lecture, or discussion
    if int(model.views[0]) == 1:
        model.T = np.random.random_sample((model.num_users, model.num_attempts,
                                           model.num_concepts))
        model.Q = np.random.random_sample((model.num_concepts, model.num_questions))
        model.bias_s = np.zeros(model.num_users)
        model.bias_t = np.zeros(model.num_attempts)
        model.bias_q = np.zeros(model.num_questions)
        model.global_bias = np.mean(model.train_data, axis=0)[3]
    else:
        raise AttributeError

    return model


def save_exp_results(model, perf_dict, data_str, course_str, model_str, fold,
                     concept_dim, lambda_t, lambda_q, lambda_bias, slr, lr, max_iter, validation):

    if not validation:
        model_dir_path = "saved_models/{}/{}/{}/fold_{}".format(data_str, course_str, model_str,
                                                                fold)
        make_dir(model_dir_path)
        para_str = "concept_{}_lt_{}_lq_{}_lbias_{}_slr_{}_" \
                   "lr_{}_max_iter_{}".format(concept_dim,lambda_t, lambda_q,lambda_bias, slr, lr, max_iter)
        model_file_path = "{}/{}_model.pkl".format(model_dir_path, para_str)
        pickle.dump(model, open(model_file_path, "wb"))

    result_dir_path = "results/{}/{}/{}".format(
        data_str, course_str, model_str
    )
    make_dir(result_dir_path)

    if validation:
        result_file_path = "{}/fold_{}_cross_val.json".format(result_dir_path, fold)
    else:
        result_file_path = "{}/fold_{}_test_results.json".format(result_dir_path, fold)

    if not os.path.exists(result_file_path):
        with open(result_file_path, "w") as f:
            pass

    result = {
        'concept_dim': concept_dim,
        'lambda_t': lambda_t,
        'lambda_q': lambda_q,
        'lambda_bias': lambda_bias,
        'student_learning_rate': slr,
        'learning_rate': lr,
        'max_iter': max_iter,
        'perf': perf_dict
    }

    output_lock.acquire()
    with open(result_file_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    output_lock.release()


def print_experiment_results(data_str, course_str, model_str, metric, num_folds=5):
    """
    find best hyperparameter via k-fold cross validation
    :param data_str:
    :param course_str:
    :param model_str:
    :param num_folds:
    :return:
    """
    combined_results = {}
    combined_detail_results = {}
    # for fold in [1,5]:
    for fold in range(1, num_folds + 1):
        output_path = "results/{}/{}/{}/fold_{}_test_results.json".format(
            data_str, course_str, model_str, fold
        )

        with open(output_path, "r") as f:
            count = 0
            for line in f:
                count += 1
                result = json.loads(line)
                key = (data_str, course_str, model_str,
                       result['concept_dim'],
                       result['lambda_t'],
                       result['lambda_q'],
                       result['lambda_bias'],
                       result['student_learning_rate'],
                       result['learning_rate'], result['max_iter'])
                perf = result["perf"]['test']
                detail_perf = result["perf"]

                if key not in combined_results:
                    combined_results[key] = {}
                combined_results[key][fold] = perf

                if key not in combined_detail_results:
                    combined_detail_results[key] = {}
                if fold not in combined_detail_results[key]:
                    combined_detail_results[key][fold] = detail_perf

    # compute the average perf over k folds on a specific metric
    for para in combined_results.keys():
        perf_list = []
        for fold in combined_results[para]:
            print(strBlue(combined_results[para][fold][metric]))
            perf_list.append(combined_results[para][fold][metric])
        avg_perf = np.mean(perf_list)
        combined_results[para] = avg_perf

    print(strRed('avg: {}'.format(avg_perf)))


    for para in combined_detail_results.keys():
        for fold in combined_detail_results[para]:
            print(strRed("\nfold, attempt, train count, train rmse, val count, val {}".format(metric)))
            for attempt in combined_detail_results[para][fold]:
                if attempt != "test":
                    train_count, train_rmse = combined_detail_results[para][fold][attempt]['train']
                    if combined_detail_results[para][fold][attempt]['test'] == {}:
                        test_count = 0
                        test_metric = 0
                    else:
                        test_count = combined_detail_results[para][fold][attempt]['test']['count']
                        test_metric = combined_detail_results[para][fold][attempt]['test'][metric]
                    print("{},{},{:.0f},{:.4f},{:.0f},{}".format(
                        fold, attempt, train_count, train_rmse, test_count, test_metric))




def run_mastery_grids():
    data_str = "mastery_grids"
    course_str = 'Quiz'
    model_str = 'fdtf'

    if course_str == "Quiz":
        concept_dim = 15
        lambda_t = 0
        lambda_q = 0.01
        lambda_bias = 0
        slr = 0.5
        lr = 0.1
        max_iter = 30

        validation = False
        metrics = ["rmse", "mae", "auc"]

        num_folds = 1
        for fold in range(1, num_folds + 1):
            log_path = "logs/{}/{}/{}/test_fold_{}/".format(data_str, course_str, model_str, fold)
            make_dir(log_path)

            para = [data_str, course_str, model_str, fold, concept_dim,
                    lambda_t, lambda_q, lambda_bias, slr, lr, max_iter]

            delimiter = '_'
            log_name = delimiter.join([str(e) for e in para[4:]])
            log_file = "{}/{}".format(log_path, log_name)
            para.append(metrics)
            para.append(log_file)
            para.append(validation)

            run_fdtf_exp(*para)


def run_morf():
    data_str = "morf"
    # course_str = 'Quiz'
    # course_str = 'Quiz_Lecture'
    # course_str = 'Quiz_Discussion'
    # course_str = 'Quiz_Lecture_Discussion'
    course_str = 'QuizIceBreaker'
    model_str = 'fdtf'

    test_range = None

    if course_str == "Quiz":
        concept_dim = 5
        lambda_t = 0.001
        lambda_q = 0
        lambda_bias = 0
        slr = 0.5
        lr = 0.1
        max_iter = 50

    if course_str == "Quiz_Lecture":
        concept_dim = 7
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        slr = 0.4
        lr = 0.1
        max_iter = 30

    if course_str == 'Quiz_Discussion':
        concept_dim = 5
        lambda_t = 0.1
        lambda_q = 0
        lambda_bias = 0
        slr = 0.7
        lr = 0.1
        max_iter = 30

    if course_str == 'Quiz_Lecture_Discussion':
        concept_dim = 9
        lambda_t = 0.1
        lambda_q = 0
        lambda_bias = 0
        slr = 0.5
        lr = 0.1
        max_iter = 30

    if course_str == 'QuizIceBreaker':
        concept_dim = 5
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0.0001
        slr = 1.0
        lr = 0.1
        max_iter = 20

        test_range = [1, 25]

    validation = False
    validation_limit = 30
    metrics = ["rmse", "mae"]

    para_list = []
    num_folds = 5

    step = 1

    if step == 1:
        for fold in range(1, num_folds + 1):
            log_path = "logs/{}/{}/{}/test_fold_{}/".format(data_str, course_str, model_str, fold)
            make_dir(log_path)

            para = [data_str, course_str, model_str, fold, concept_dim,
                    lambda_t, lambda_q, lambda_bias, slr, lr, max_iter]

            delimiter = '_'
            log_name = delimiter.join([str(e) for e in para[4:]])
            log_file = "{}/{}".format(log_path, log_name)
            para.append(metrics)
            para.append(log_file)
            para.append(validation)
            para.append(validation_limit)
            para.append(test_range)


            # run_fdtf_exp(*para)
            para_list.append(para)
        pool = Pool(processes=5)
        pool.starmap(run_fdtf_exp, para_list)
        pool.close()

    if step == 2:
        print_experiment_results(data_str, course_str, model_str, metric = "rmse", num_folds = 5)


def run_laura():
    data_str = "laura"
    course_str = 'QuizIceBreaker'
    model_str = 'fdtf'


    if course_str == 'QuizIceBreaker':
        concept_dim = 7
        lambda_t = 0.001
        lambda_q = 0
        lambda_bias = 0.001
        slr = 0.5
        lr = 0.1
        max_iter = 20

        test_range = [1, 50]

    validation = False
    validation_limit = 30
    metrics = ["rmse", "mae"]

    para_list = []
    num_folds = 5

    step = 1

    if step == 1:
        for fold in range(1, num_folds + 1):
            log_path = "logs/{}/{}/{}/test_fold_{}/".format(data_str, course_str, model_str, fold)
            make_dir(log_path)

            para = [data_str, course_str, model_str, fold, concept_dim,
                    lambda_t, lambda_q, lambda_bias, slr, lr, max_iter]

            delimiter = '_'
            log_name = delimiter.join([str(e) for e in para[4:]])
            log_file = "{}/{}".format(log_path, log_name)
            para.append(metrics)
            para.append(log_file)
            para.append(validation)
            para.append(validation_limit)
            para.append(test_range)


            # run_fdtf_exp(*para)
            para_list.append(para)
        pool = Pool(processes=5)
        pool.starmap(run_fdtf_exp, para_list)
        pool.close()

    if step == 2:
        print_experiment_results(data_str, course_str, model_str, metric = "rmse", num_folds = 5)


if __name__ == '__main__':
    # run_mastery_grids()
    run_morf()
    # run_laura()