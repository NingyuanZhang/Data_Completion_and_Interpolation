import numpy as np
from utils import GD_Optimizer
from dataloader import DataLoader
from model_np import Model
import utils
import os

lambda1 = 0.0001    # l1 weight
theta = 0.01        # model weight initialization multiplier
lr = 1e-3
print_every = 10000
valid_every = 10000

def train(train_x, train_target, model):
    prediction = model.predict(train_x)
    loss = utils.cross_entropy(prediction, train_target)
    l1 = 0.0
    for param in model.parameters(flatten=True):
        l1 += np.abs(param)
    loss = loss + lambda1 * l1
    return loss

def validation(valid_x, valid_target, model):
    prediction = model.predict(valid_x)
    loss = utils.cross_entropy(prediction, valid_target)
    return loss

def test(test_x, test_target, model):
    prediction = model.predict(test_x)
    loss = utils.cross_entropy(prediction, test_target)
    return loss

def runner(data_loader, train_df, valid_df, test_df, num_classes, label, epochs=50):
    ###########################
    # Prepare dataset
    ###########################
    train_x, train_target = data_loader.process_np(train_df, label)
    valid_x, valid_target = data_loader.process_np(valid_df, label)
    test_x, test_target = data_loader.process_np(test_df, label)

    ###########################
    # setup model and optimizer
    ###########################
    bias = True
    num_features = train_x.shape[1]
    model = Model(num_features, num_classes, bias=bias, theta=theta)
    if bias is True:
        data_x = np.append(train_x, np.ones((train_x.shape[0], 1)), axis=1)
    else:
        data_x = train_x
    optimizer = GD_Optimizer(model.parameters(), data_x, train_target, lr=lr, l1_weight=lambda1)

    ###########################
    # Process start
    ###########################
    min_valid_loss = 100000.
    saved_test_loss = 0.
    best_epoch = 0
    for epoch in range(epochs):
        # Train
        optimizer.zero_grad()
        train_loss = train(train_x, train_target, model)
        optimizer.backward()
        optimizer.step()
        # Validation
        valid_loss = validation(valid_x, valid_target, model)
        # Test
        test_loss = test(test_x, test_target, model)

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            saved_test_loss = test_loss
            model_name = label + '_model.pt'
            model_path = os.path.join('model', model_name)
            # save
            utils.save_model(model, model_path)

        # early stop
        if epoch - best_epoch > 20000:
            print('train complete...')
            break

        if epoch % print_every == 0:
            print('Train loss: %.4f, epoch %d' % (train_loss, epoch))
            print('Validation loss: %.4f, epoch %d' % (valid_loss, epoch))
            print('Test loss: %.4f, epoch %d' % (test_loss, epoch))
            print('---------------------------------------------')
            print('Current min valid loss: %.4f' % min_valid_loss)
            print('Corresponding test loss: %.4f' % saved_test_loss)
            print('---------------------------------------------')
            print('\n')

if __name__ == '__main__':
    # prepare parameters, you can use command line to receive these arguments
    path = '../data/data_all.csv'
    header_file_path = '../data/new_features.csv'
    # Load data and split data
    dl = DataLoader(path, header_file_path)
    x, y = dl.get_xy_headers()
    y_classes = dl.get_y_class_number()
    # build column list for all models and train models
    for idx, label in enumerate(y):
        print('Current: {}/{}'.format(idx + 1, len(y)))
        print(label)
        columns = [x for x in x]
        columns.append(label)
        num_class = y_classes[label]
        train_df, valid_df, test_df = dl.data_split(train_ratio=0.8, col_list=columns)
        runner(dl, train_df, valid_df, test_df, num_classes=num_class, label=label, epochs=100000)