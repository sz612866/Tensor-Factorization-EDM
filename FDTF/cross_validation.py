from utils import make_dir, strRed, strBlue
from run import run_fdtf_exp
import numpy as np
import os
import json
import datetime
from multiprocessing import Pool, Lock

output_lock = Lock()


def check_progress(data_str, course_num, model_str, fold, restart=False):
    """
    check finished runs of hyper-parameters
    :param data_str:
    :param course_num:
    :param model_str:
    :param fold:
    :param restart:
    :return:
    """
    output_path = "results/{}/{}/{}/fold_{}_cross_val.json".format(
        data_str, course_num, model_str, fold
    )

    progress_dict = {}
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            pass
        return progress_dict

    if restart:
        today = datetime.date.today()
        os.rename(output_path, "{}(renamed_on_{})".format(output_path, today))
        print(strRed("restart hyperparameter tuning on fold: {}".format(fold)))
        with open(output_path, "w") as f:
            pass
        return progress_dict

    with open(output_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            result = json.loads(line)
            key = (result['concept_dim'],
                   result['lambda_t'],
                   result['lambda_q'],
                   result['lambda_bias'],
                   result['student_learning_rate'],
                   result['learning_rate'],
                   result['max_iter'])
            if key not in progress_dict:
                progress_dict[key] = True
    return progress_dict



def morf_hyperparameter_tuning(data_str, course_str, model_str, metrics, fold, num_proc):
    """
    grid search the optimal hyperparameters that achieve best 5-fold cross-validation performance
    :param metrics:
    :param data_str: name of dataset
    :param course_str: course number
    :param model_str: model name
    :param fold:
    :param num_proc:
    :return: the best hyperparameters and corresponding 5 fold cross validation performance
    """

    validation = True
    log_path = "logs/{}/{}/{}/val_fold_{}/".format(data_str, course_str, model_str, fold)
    make_dir(log_path)

    if course_str == 'Quiz':
        # check processed hyper-parameters
        restart = False
        # restart = False
        progress_dict = check_progress(data_str, course_str, model_str, fold, restart)

        lambda_bias = 0
        lr = 0.1


        para_list = []
        remaining = 0
        for concept_dim in [3, 5, 7, 9]:
            for lambda_t in [0, 0.001, 0.01, 0.1]:
                for lambda_q in [0, 0.001, 0.01, 0.1]:
                    for lambda_bias in [0, 0.001, 0.01, 0.1]:
                        for slr in [0.1 * k for k in range(1, 11)]:
                            for max_iter in [30, 50]:
                                # for max_iter in [50]:
                                para = (data_str, course_str, model_str, fold,
                                        concept_dim, lambda_t, lambda_q,
                                        lambda_bias, slr, lr, max_iter)
                                if para[4:] in progress_dict:
                                    continue
                                else:
                                    remaining += 1
                                delimiter = '_'
                                log_name = delimiter.join([str(e) for e in para[4:]])
                                log_file = "{}/{}".format(log_path, log_name)
                                para = list(para)
                                para.append(metrics)
                                para.append(log_file)
                                para.append(validation)
                                para_list.append(para)
        print("fold {} experiment configuration: [# of cores: {}, completed: {}, remaining runs: "
              "{}]".format(fold, num_proc, len(progress_dict), remaining))
        pool = Pool(processes=num_proc)
        pool.starmap(run_fdtf_exp, para_list)
        pool.close()


def find_best_hyperparameters(data_str, course_str, model_str, metric, num_folds=5):
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
    for fold in range(1, num_folds + 1):
        output_path = "results/{}/{}/{}/fold_{}_cross_val.json".format(
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
                perf = result["perf"]['val']
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
            perf_list.append(combined_results[para][fold][metric])
        avg_perf = np.mean(perf_list)
        combined_results[para] = avg_perf

    # print the sorted results
    print(strRed('data_str, course_num, model_str, concept_dim, lambda_t, '
                 'lambda_q, lambda_bias, slr, lr, max_iter'))
    for (key, val) in sorted(combined_results.items(), key=lambda x: x[1]):
        for k in key:
            print("{},".format(k), end="")
        print(strBlue("{:.5f}".format(val)))

    if metric in ["rmse", "mae"]:
        # find the best perf and corresponding detail results
        best_key, best_perf = sorted(combined_results.items(), key=lambda x: x[1])[0]
        best_detail_perf = combined_detail_results[best_key]
    elif metric in ["auc"]:
        best_key, best_perf = sorted(combined_results.items(), key=lambda x: x[1])[-1]
        best_detail_perf = combined_detail_results[best_key]
    else:
        raise AttributeError

    print(strRed("\nbest hyperparameters:"))
    print(' data_str={}\n course_str={}\n model_str={}\n concept_dim={}\n '
          'lambda_t={}\n lambda_q={}\n lambda_bias={}\n '
          'slr={}\n lr={}\n max_iter={}'.format(*best_key))

    for fold in best_detail_perf.keys():
        print(strRed("\nfold, attempt, train count, train rmse, val count, val {}".format(metric)))
        for attempt in best_detail_perf[fold]:
            if attempt != "val":
                train_count, train_rmse = best_detail_perf[fold][attempt]['train']
                val_count = best_detail_perf[fold][attempt]['val']['count']
                val_metric = best_detail_perf[fold][attempt]['val'][metric]
                print("{},{},{:.0f},{:.4f},{:.0f},{}".format(
                    fold, attempt, train_count, train_rmse, val_count, val_metric))

        print(strRed("fold, val count, {}".format(metric)))
        val_perf = best_detail_perf[fold]["val"]
        print(strBlue("{}, {}, {:.4f}".format(fold, val_perf["count"], val_perf[metric])))



def run_morf():
    data_str = "morf"
    course_str = "Quiz"
    model_str = "fdtf"
    metrics = ["rmse", "mae"]


    # step 1. grid search of hyperparameters
    num_proc = 64
    fold = 5
    morf_hyperparameter_tuning(data_str, course_str, model_str, metrics, fold, num_proc)

    # step 2. run the find_best_hyperparameter function
    # find_best_hyperparameters(data_str, course_str, model_str, metric="rmse", num_folds=5)



if __name__ == '__main__':
    run_morf()
