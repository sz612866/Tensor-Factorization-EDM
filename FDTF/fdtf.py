from utils import *
import numpy as np
from numpy import linalg as LA
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import warnings

warnings.filterwarnings("error")


class MultiView(object):

    def __init__(self, config):
        """
        :param config:
        :var
        """

        np.random.seed(1)

        if self.log_file:
            self.logger = create_logger(self.log_file)
        self.views = config['views']
        self.train_data = config['train']

        self.num_users = config['num_users']
        self.num_attempts = config['num_attempts']
        self.num_concepts = config['num_concepts']
        self.num_questions = config['num_questions']
        self.lambda_t = config['lambda_t']
        self.lambda_q = config['lambda_q']
        self.lambda_bias = config['lambda_bias']

        self.lr = config['lr']
        self.max_iter = config['max_iter']
        self.tol = config['tol']

        self.metrics = config['metrics']

        self.use_bias_t = False
        self.use_global_bias = True

        self.binarized_question = True

        self.current_test_attempt = None
        self.test_obs_list = []
        self.test_pred_list = []

        self.T = np.random.random_sample((self.num_users, self.num_attempts,
                                          self.num_concepts))
        self.Q = np.random.random_sample((self.num_concepts, self.num_questions))

        self.bias_s = np.zeros(self.num_users)
        self.bias_t = np.zeros(self.num_attempts)
        self.bias_q = np.zeros(self.num_questions)

        self.global_bias = np.mean(self.train_data, axis=0)[3]

    def __getstate__(self):
        """
        since the logger cannot be pickled, to avoid the pickle error, we should add this
        :return:
        """
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


    def _get_question_prediction(self, student, attempt, question):
        """
        predict value at tensor Y[attempt, student, question]
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor Y[attempt, student, question]
        """
        pred = np.dot(np.dot(self.T[student, attempt, :]), self.Q[:, question])
        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_q[question] + \
                        self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_q[question]
        else:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_q[question] + self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_q[question]

        if self.binarized_question:
            pred = sigmoid(pred)
        return pred


    def _get_loss(self):
        """
        override the function in super class
        compute the loss, which is RMSE of observed records
        :return: loss
        """
        loss, square_loss, reg_bias = 0., 0., 0.
        square_loss_q = 0., 0., 0.
        q_count = 0., 0., 0.

        for (student, attempt, question, obs, resource) in self.train_data:
            pred = self._get_question_prediction(student, attempt, question)
            square_loss_q += (obs - pred) ** 2
            q_count += 1

        print("square loss {}".format(square_loss_q))

        reg_T = LA.norm(self.T) ** 2  # regularization on tensor T
        reg_Q = LA.norm(self.Q) ** 2  # regularization on matrix Q

        reg_features = self.lambda_q * reg_Q + self.lambda_t * reg_T

        q_rmse = np.sqrt(square_loss_q / q_count) if q_count != 0 else 0.

        if self.lambda_bias:
            if self.use_bias_t:
                reg_bias = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_t) ** 2 +
                        LA.norm(self.bias_q) ** 2)
            else:
                reg_bias = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_q) ** 2)
 
        loss = square_loss_q + reg_features + reg_bias
        return loss, q_count, q_rmse, reg_features, reg_bias


    def _grad_T_ij(self, student, attempt, index, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific student j's knowledge at
        a specific attempt i: T_{i,j,:},
        :param attempt: index
        :param student: index
        :param obs: observation
        :return:
        """

        grad = np.zeros_like(self.T[student, attempt, :])

        if obs is not None:
            pred = self._get_question_prediction(student, attempt, index)
            if self.binarized_question:
                grad = -2. * (obs - pred) * pred * (1. - pred) * self.Q[:, index] +\
                       2. * self.lambda_t * self.T[student, attempt, :]
            else:
                grad = -2. * (obs - pred) * self.Q[:, index] + \
                       2. * self.lambda_t * self.T[student, attempt, :]

        return grad



    def _grad_Q_k(self, student, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question association
        of a question in Q-matrix,
        :param attempt: index
        :param student:  index
        :param question:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """

        grad = np.zeros_like(self.Q[:, question])
        if obs is not None:
            pred = self._get_question_prediction(student, attempt, question)
            if self.binarized_question:
                grad = -2. * (obs - pred) * pred * (1. - pred) * self.T[student, attempt, :] + \
                       2. * self.lambda_q * self.Q[:, question]
            else:
                grad = -2. * (obs - pred) * self.T[student, attempt, :] + \
                       2. * self.lambda_q * self.Q[:, question]

        return grad

    def _grad_bias_s(self, student, attempt, material, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param attempt:
        :param student:
        :param material: material material of that resource
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, attempt, material)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred) + \
                        2.0 * self.lambda_bias * self.bias_s[student]
            else:
                grad -= 2. * (obs - pred) + 2.0 * self.lambda_bias * self.bias_s[student]

        return grad



    def _grad_bias_t(self, student, attempt, material, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific bias_a
        :param attempt:
        :param student:
        :param material: material material of that resource
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, attempt, material)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred) + \
                        2.0 * self.lambda_bias * self.bias_t[attempt]
            else:
                grad -= 2. * (obs - pred) + 2.0 * self.lambda_bias * self.bias_t[attempt]
        return grad



    def _grad_bias_q(self, student, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, attempt, question)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred) + \
                        2. * self.lambda_bias * self.bias_q[question]
            else:
                grad -= 2. * (obs - pred) + 2. * self.lambda_bias * self.bias_q[question]

        return grad


