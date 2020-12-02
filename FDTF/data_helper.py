# some helper functions for generating suitable data or configuration on running experiments
from utils import *
import gensim
import nltk

def fdtf_config(data_str, course_str, views, concept_dim, fold, lambda_t, lambda_q, lambda_bias, slr, lr, max_iter,
                metrics, log_file, validation, validation_limit=30, test_range = None):

    """
    generate model configurations for training and testing
    such as initialization of each parameters and hyperparameters
    :return: config dict
    """

    with open('data/{}/{}/{}_train_val_test.pkl'.format(data_str, course_str, fold), 'rb') as f:
        data = pickle.load(f)


    # learning materials for concatenation
    num_questions = data['num_quizzes']
    if views == "110": # learning materials for concatenation of Quiz_Lecture
        num_questions += data['num_lectures']
    if views == "101": # learning materials for concatenation of Quiz_Discussion
        num_questions += data['num_discussions']
    if views == "111": # learning materials for concatenation of Quiz_Lecture_Discussion
        num_questions += data['num_lectures'] + data['num_discussions']

    config = {
        'views': views,
        'num_users': data['num_users'],
        'num_attempts': data['num_attempts'],
        'num_questions': num_questions,
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

    max_stu_attempt = {}

    # generate config, train_set, test_set for general train and test
    train_data = []
    for (student, attempt, question, score, resource) in data['train']:
        # if it is for validation, we only use the first 30 attempts for cross validation to
        # do the hyperparameter tuning

        if views == "110" and int(resource) == 1: #concatenation for Lecture in  Quiz_Lecture
            question = int(question) + data['num_quizzes']
        if views == "101" and int(resource) == 1: #concatenation for Discussion in  Quiz_Discussion
            question = int(question) + data['num_quizzes']
        if views == "111" and int(resource) == 1: #concatenation for Lecture in Quiz_Lecture_Discussion
            question = int(question) + data['num_quizzes']
        if views == "111" and int(resource) == 2: #concatenation for Discussion in Quiz_Lecture_Discussion
            question = int(question) + data['num_quizzes'] + data['num_lectures']

        if validation:
            if attempt < validation_limit:
                train_data.append(
                    (int(student), int(attempt), int(question), float(score), int(resource))
                )

                max_stu_attempt[int(student)] = int(attempt)
        # elif test_range:
        #     if attempt < test_range[1]:
        #         train_data.append(
        #             (int(student), int(attempt), int(question), float(score), int(resource))
        #         )
        #         max_stu_attempt[int(student)] = int(attempt)
        else:
            train_data.append(
                (int(student), int(attempt), int(question), float(score), int(resource))
            )

    config['train'] = train_data

    val_data = []
    for (student, attempt, question, score, resource) in data['val']:
        if views == "110" and int(resource) == 1: #concatenation for Lecture in  Quiz_Lecture
            question = int(question) + data['num_quizzes']
        if views == "101" and int(resource) == 1: #concatenation for Discussion in  Quiz_Discussion
            question = int(question) + data['num_quizzes']
        if views == "111" and int(resource) == 1: #concatenation for Lecture in Quiz_Lecture_Discussion
            question = int(question) + data['num_quizzes']
        if views == "111" and int(resource) == 2: #concatenation for Discussion in Quiz_Lecture_Discussion
            question = int(question) + data['num_quizzes'] + data['num_lectures']
        if validation:
            if attempt < validation_limit:
                val_data.append(
                    (int(student), int(attempt), int(question), float(score), int(resource))
                )

        # elif test_range:
        #     if attempt > test_range[0] and attempt < test_range[1]:
        #         val_data.append(
        #             (int(student), int(attempt), int(question), float(score), int(resource))
        #         )
        #         max_stu_attempt[int(student)] = int(attempt)

        else:
            val_data.append(
                (int(student), int(attempt), int(question), float(score), int(resource))
            )

    config['val'] = val_data

    test_set = []
    test_users = {}
    for (student, attempt, question, score, resource) in data['test']:
        if views == "110" and int(resource) == 1: #concatenation for Lecture in  Quiz_Lecture
            question = int(question) + data['num_quizzes']
        if views == "101" and int(resource) == 1: #concatenation for Discussion in  Quiz_Discussion
            question = int(question) + data['num_quizzes']
        if views == "111" and int(resource) == 1: #concatenation for Lecture in Quiz_Lecture_Discussion
            question = int(question) + data['num_quizzes']
        if views == "111" and int(resource) == 2: #concatenation for Discussion in Quiz_Lecture_Discussion
            question = int(question) + data['num_quizzes'] + data['num_lectures']
        if validation:
            if attempt < validation_limit:
                test_set.append(
                    (int(student), int(attempt), int(question), float(score), int(resource))
                )

        elif test_range:
            if attempt > test_range[0] and attempt < test_range[1]:
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

    # get each student's maximum student attempt
    for stu, att, question, score, resource in sorted(train_data + test_set + val_data, key = lambda x:x[1]):
        max_stu_attempt[int(stu)] = int(att)

    config['max_stu_attempt'] = max_stu_attempt

    return config




