from algorithms import Algorithm_1
from algorithms import Algorithm_2
from algorithms import Algorithm_3
from data import DataProvider
from data import DataPreprocessing
import json
from matplotlib import pyplot as plt
import pandas as pd


def main():
    print("Starting...")
    print("")

    # read config.json
    with open('config/config.json') as file:
        config = json.load(file)

    # Get Parameters from config file
    algo1_lr = config["Algorithm 1"]["learning_rate"]
    algo1_epochs = config["Algorithm 1"]["n_epochs"]
    algo1_opt = config["Algorithm 1"]["opt"]
    algo1_min = config["Algorithm 1"]["min"]
    algo1_max = config["Algorithm 1"]["max"]
    algo1_step = config["Algorithm 1"]["step"]

    algo2_lr = config["Algorithm 2"]["learning_rate"]
    algo2_epochs = config["Algorithm 2"]["n_epochs"]
    algo2_opt = config["Algorithm 2"]["opt"]
    algo2_min = config["Algorithm 2"]["min"]
    algo2_max = config["Algorithm 2"]["max"]
    algo2_step = config["Algorithm 2"]["step"]

    algo3_lr = config["Algorithm 3"]["learning_rate"]
    algo3_epochs = config["Algorithm 3"]["n_epochs"]
    algo3_opt = config["Algorithm 3"]["opt"]
    algo3_min = config["Algorithm 3"]["min"]
    algo3_max = config["Algorithm 3"]["max"]
    algo3_step = config["Algorithm 3"]["step"]

    # Get Data (Preprocessed)
    Preprocessor = DataPreprocessing.DataPreprocessing()
    Preprocessor.clean_text()

    Provider = DataProvider.DataProvider()
    train_data, test_data = Provider.import_data()

    ################################################################################################################
    # Convolutional Neural Network
    ################################################################################################################
    ConvNN_training = []
    ConvNN_test = []

    for i in range(algo1_min, algo1_max, algo1_step):
        model = Algorithm_1.TensorFlow_CNN(train_data, test_data, algo1_lr, algo1_epochs, algo1_opt, i)
        duration_train, acc_train, n_params = model.train()
        duration_test, acc_test = model.test()

        ConvNN_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'parameter': n_params,
             'Run': i
             }
        )
        ConvNN_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    ConvNN_training_df = pd.DataFrame(ConvNN_training)
    ConvNN_test_df = pd.DataFrame(ConvNN_test)

    ################################################################################################################
    # Recurrent Neural Network
    ################################################################################################################
    DBOW_training = []
    DBOW_test = []

    for i in range(algo2_min, algo2_max, algo2_step):
        model = Algorithm_2.RNNEmbeddingLayer(train_data, test_data, algo2_lr, algo2_epochs, algo2_opt, i)
        duration_train, acc_train, n_params = model.train()
        duration_test, acc_test = model.test()

        DBOW_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'parameter': n_params,
             'Run': i
             }
        )
        DBOW_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    DBOW_training_df = pd.DataFrame(DBOW_training)
    DBOW_test_df = pd.DataFrame(DBOW_test)

    ################################################################################################################
    # Embedding Layer + Neural Network
    ################################################################################################################
    NeuralNetwork_training = []
    NeuralNetwork_test = []

    for i in range(algo3_min, algo3_max, algo3_step):
        model = Algorithm_3.NeuralNetworkEmbeddingLayer(train_data, test_data, algo3_lr, algo3_epochs, algo3_opt, i)
        duration_train, acc_train, n_params = model.train()
        duration_test, acc_test = model.test()

        NeuralNetwork_training.append(
            {'accuracy': acc_train,
             'duration': duration_train,
             'parameter': n_params,
             'Run': i
             }
        )
        NeuralNetwork_test.append(
            {'accuracy': acc_test,
             'duration': duration_test,
             'Run': i
             }
        )
        model = None

    NeuralNetwork_training_df = pd.DataFrame(NeuralNetwork_training)
    NeuralNetwork_test_df = pd.DataFrame(NeuralNetwork_test)

    ################################################################################################################
    # Evaluation
    ################################################################################################################
    px = 1 / plt.rcParams['figure.dpi']

    fig = plt.figure(figsize=(1000 * px, 700 * px))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_xlabel('Model-Performance (Accuracy in %)', fontsize=10)
    ax1.set_ylabel('Duration [in seconds]', fontsize=10)
    ax2.set_xlabel('Model-Performance (Accuracy in %)', fontsize=10)
    ax2.set_ylabel('Duration [in seconds]', fontsize=10)
    ax1.plot(ConvNN_test_df["accuracy"], ConvNN_training_df["duration"], '-o', c='blue', alpha=0.6, markersize=4)
    ax2.plot(ConvNN_test_df["accuracy"], ConvNN_test_df["duration"], '-o', c='blue', alpha=0.6, markersize=4)
    ax1.plot(DBOW_test_df["accuracy"], DBOW_training_df["duration"], '-o', c='green', alpha=0.6,
             markersize=4)
    ax2.plot(DBOW_test_df["accuracy"], DBOW_test_df["duration"], '-o', c='green', alpha=0.6,
             markersize=4)
    ax1.plot(NeuralNetwork_test_df["accuracy"], NeuralNetwork_training_df["duration"], '-o', c='red', alpha=0.6,
             markersize=4)
    ax2.plot(NeuralNetwork_test_df["accuracy"], NeuralNetwork_test_df["duration"], '-o', c='red', alpha=0.6, markersize=4)
    ax1.title.set_text('Training')
    ax2.title.set_text('Inference')
    plt.legend(["Universal Sentence Encoder + ANN", "Embedding Layer + RNN", "Embedding Layer + ANN"],
               loc='lower center', ncol=3, bbox_transform=fig.transFigure, bbox_to_anchor=(0.5, 0))
    plt.savefig('plots/Algorithms_Evaluation.png', dpi=600)
    plt.clf()
    print("Evaluation Plot saved...")
    print("")

    plt.figure(figsize=(650 * px, 400 * px))
    plt.plot(ConvNN_training_df["accuracy"], ConvNN_training_df["parameter"], '-o', c='blue', alpha=0.6)
    plt.plot(DBOW_training_df["accuracy"], DBOW_training_df["parameter"], '-o', c='green', alpha=0.6)
    plt.plot(NeuralNetwork_training_df["accuracy"], NeuralNetwork_training_df["parameter"], '-o', c='red', alpha=0.6)
    plt.xlabel('Accuracy [in %]', fontsize=10)
    plt.ylabel('Number of Parameter', fontsize=10)
    plt.legend(["Universal Sentence Encoder + Neural Network",
                "Embedding Layer + Recurrent Neural Network",
                "Embedding Layer + Neural Network"])
    # plt.yscale('log')
    plt.savefig('plots/Number_of_Parameter.png', dpi=300)
    plt.clf()
    print("Parameter Plot saved...")
    print("")


if __name__ == "__main__":
    main()
