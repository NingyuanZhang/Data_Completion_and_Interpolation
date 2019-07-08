from dataloader import DataLoader
import utils
import numpy as np

def test(test_x, model):
    prediction = model.predict(test_x)
    return prediction

def runner(data_loader, test_df, num_classes):
    # split data
    test_x, test_target = data_loader.process_np(test_df)
    # Setup model
    model = utils.load_model()
    print(model.parameters())
    print(test_target)
    probability = test(test_x, model)

    # predicted_probs = np.amax(probability, axis=1)
    print(probability)
    predicted_class = np.argmax(probability, axis=1)
    print('-------- Prediction Accuracy ---------')
    predicted_class = predicted_class.flatten()
    print(predicted_class)
    assert (len(predicted_class) == len(test_target))
    count = 0
    for i, value in enumerate(predicted_class):
        if value == test_target[i]:
            count += 1
    accuracy = count / len(predicted_class)
    print(accuracy)

if __name__ == '__main__':
    # prepare parameters, you can use command line to receive these arguments
    path = 'out2.csv'
    label = 'intrinsic_15'
    # prepare columns to be used for training
    columns = []
    for i in range(1, 16):
        if i < 10:
            s = 'intrinsic_0' + str(i)
            columns.append(s)
        else:
            s = 'intrinsic_' + str(i)
            columns.append(s)
    # Load data and split data
    dl = DataLoader(path, label=label, random_seed=2012)
    _, _, test_df = dl.data_split(train_ratio=0.8, col_list=columns)
    # Start program
    runner(dl, test_df, num_classes=4)