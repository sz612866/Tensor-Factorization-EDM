from data_helper import fdtf_config
from fdtf import FDTF
from utils import make_dir
import json
import os
import numpy as np
import pickle
from multiprocessing import Lock

output_lock = Lock()


def run_fdtf_exp(data_str, course_str, model_str, fold, concept_dim, lambda_t,
                 lambda_q, lambda_bias, slr, lr, max_iter, metrics, log_file, validation):
    """
    cross validation on the first fold of dataset to tune the hyper-parameters,
    then we will use those best hyper-parameters evaluate the performance on all folds
    """

    if course_str == "Quiz":
        views = "100"
    elif course_str == "Lecture":
        views = "010"
    elif course_str == "Discussion":
        views = "001"
    else:
        raise IOError

    model_config = fdtf_config(
        data_str, course_str, views, concept_dim, fold, lambda_t, lambda_q,
        lambda_bias, slr, lr, max_iter, metrics, log_file, validation)


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
    course_str = 'Quiz'
    model_str = 'fdtf'

    if course_str == "Quiz":
        concept_dim = 5
        lambda_t = 0.001
        lambda_q = 0
        lambda_bias = 0
        slr = 0.5
        lr = 0.1
        max_iter = 50

        validation = False
        metrics = ["rmse", "mae"]

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


if __name__ == '__main__':
    # run_mastery_grids()
    run_morf()