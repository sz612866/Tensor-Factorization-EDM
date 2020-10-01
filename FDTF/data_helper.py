# some helper functions for generating suitable data or configuration on running experiments
from utils import *
import gensim
import nltk

def fdtf_config(data_str, course_str, views, concept_dim, fold, lambda_t, lambda_q, lambda_bias, slr, lr, max_iter,
                metrics, log_file, validation, validation_limit=30):

    """
    generate model configurations for training and testing
    such as initialization of each parameters and hyperparameters
    :return: config dict
    """

    with open('data/{}/{}/{}_train_val_test.pkl'.format(data_str, course_str, fold), 'rb') as f:
        data = pickle.load(f)

    config = {
        'views': views,
        'num_users': data['num_users'],
        'num_attempts': data['num_attempts'],
        'num_questions': data['num_quizzes'],
        'num_concepts': concept_dim,
        'lambda_t': lambda_t,
        'lambda_q': lambda_q,
        'lambda_bias': lambda_bias,
        'lr': lr,
        'max_iter': max_iter,
        'tol': 1e-3,
        'slr': slr,
        'metrics': metrics,
        'log_file': log_file
    }

    # generate config, train_set, test_set for general train and test
    train_data = []
    for (student, attempt, question, score, resource) in data['train']:
        # if it is for validation, we only use the first 30 attempts for cross validation to
        # do the hyperparameter tuning
        if validation:
            if attempt < validation_limit:
                train_data.append(
                    (int(student), int(attempt), int(question), float(score), int(resource))
                )
        else:
            train_data.append(
                (int(student), int(attempt), int(question), float(score), int(resource))
            )
    config['train'] = train_data

    val_data = []
    for (student, attempt, question, score, resource) in data['val']:
        if validation:
            if attempt < validation_limit:
                val_data.append(
                    (int(student), int(attempt), int(question), float(score), int(resource))
                )
        else:
            val_data.append(
                (int(student), int(attempt), int(question), float(score), int(resource))
            )
    config['val'] = val_data

    test_set = []
    test_users = {}
    for (student, attempt, question, score, resource) in data['test']:
        if validation:
            if attempt < validation_limit:
                test_set.append(
                    (int(student), int(attempt), int(question), float(score), int(resource))
                )
        else:
            test_set.append(
                (int(student), int(attempt), int(question), float(score), int(resource))
            )
        if student not in test_users:
            test_users[student] = 1
    config['test'] = test_set
    if validation:
        config['num_attempts'] = validation_limit

    return config




