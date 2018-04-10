import numpy as np
from keras.models import Sequential
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report, confusion_matrix, r2_score, roc_curve, auc
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
plt.style.use('fivethirtyeight')

class ModelEvaluator():

    def __init__(self, model, X_test, y_test, batch_size, generate_plots=True, binary_classification=True):
        self.model = model
        self.y_test = y_test
        self.binary_classification = binary_classification


        # Evaluate and Predict
        self.score, self.acc = self.model.Evaluate(X_test, y_test, batch_size)
        self.Y_score, self.Y_predict, self.Y_true = self.model.Predict(X_test, y_test)


        # Generate Metrics

        self.GeneratePerformanceSummary()

        if (generate_plots):
            self.GenerateConfusionMatrix()
            self.GenerateROCPlot()
            self.GeneratePrecisionRecallPlot()
            self.GenerateReliabilityPlot()
            self.GenerateCumulativeGainPlot()
            self.GenerateLiftPlot()




    def GeneratePerformanceSummary(self):

        print('--- Performance Summary ---')

        print('Classification Report: \n')
        print(classification_report(self.y_test, self.Y_predict, digits=5))

        print('R2: ', r2_score(self.y_test, self.Y_predict))

        if self.binary_classification:
            print('ROC/AUC Score: ', roc_auc_score(self.y_test, self.Y_score))


    def GenerateROCPlot(self):

        y_probas = np.concatenate((1 - self.Y_score, self.Y_score), axis=1)
        skplt.metrics.plot_roc_curve(self.y_test, y_probas)
        plt.show()

    def GeneratePrecisionRecallPlot(self):

        y_probas = np.concatenate((1 - self.Y_score, self.Y_score), axis=1)
        skplt.metrics.plot_precision_recall_curve(self.y_test, y_probas)
        plt.show()

    def GenerateCumulativeGainPlot(self):

        y_probas = np.concatenate((1 - self.Y_score, self.Y_score), axis=1)
        skplt.metrics.plot_cumulative_gain(self.y_test, y_probas)
        plt.show()


    def GenerateLiftPlot(self):
        y_probas = np.concatenate((1 - self.Y_score, self.Y_score), axis=1)
        skplt.metrics.plot_lift_curve(self.y_test, y_probas)
        plt.show()

    def GenerateReliabilityPlot(self):
        y_probas = np.concatenate((1 - self.Y_score, self.Y_score), axis=1)
        probas_list = [y_probas ]
        y_true = list(self.y_test)
        skplt.metrics.plot_calibration_curve(self.y_test, probas_list)
        plt.show()


    def GenerateConfusionMatrix(self):

        skplt.metrics.plot_confusion_matrix(self.y_test, self.Y_predict)
        plt.show()

        # print('Confusion Matrix: \n')
        # conf_matrix = confusion_matrix(self.y_test, self.Y_predict)
        # print(conf_matrix)
        #
        # conf_matrix_data = {
        #     1: {
        #         'matrix': conf_matrix,
        #         'title': 'Confusion Matrix of Classifier',
        #     },
        # }
        #
        # plt.suptitle('Confusion Matrix of Classifier')
        # for ii, values in conf_matrix_data.items():
        #     matrix = values['matrix']
        #     title = values['title']
        #     plt.plot(3, 3, ii)  # starts from 1
        #     plt.title(title);
        #     sns.heatmap(matrix, annot=True, fmt='');
        #
        # plt.show()
        #
        #
        # print('---------------------------')


    def PlotTrainingPerformance(self, history):

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


