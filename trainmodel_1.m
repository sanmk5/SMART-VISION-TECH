% Define paths for the dataset
datasetPath = "D:\FLIPKART EVENT\DATASET"; % Replace with your dataset path
outputModelPath = 'D:\FLIPKART EVENT\5brandtrain.mat'; % Path to save the trained model

% Load dataset
imdsTrain = shuffle(imdsTrain);
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageDataAugmenter);
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); % Use folder names as labels (brand names)

% Split dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Define network architecture
inputSize =[128, 128, 3]; % For custom CNN input size (same as ResNet50)

% Define a simple CNN architecture
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_1') % First convolution layer
    batchNormalizationLayer('Name', 'bn_1') % Batch normalization
    reluLayer('Name', 'relu_1') % ReLU activation
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1') % Max pooling layer
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_2') % Second convolution layer
    batchNormalizationLayer('Name', 'bn_2') % Batch normalization
    reluLayer('Name', 'relu_2') % ReLU activation
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2') % Max pooling layer
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv_3') % Third convolution layer
    batchNormalizationLayer('Name', 'bn_3') % Batch normalization
    reluLayer('Name', 'relu_3') % ReLU activation
    
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3') % Max pooling layer
    
    fullyConnectedLayer(128, 'Name', 'fc_1') % Fully connected layer
    reluLayer('Name', 'relu_4') % ReLU activation
    
    fullyConnectedLayer(numel(categories(imds.Labels)), 'Name', 'fc_2') % Output layer (number of classes)
    softmaxLayer('Name', 'softmax') % Softmax for classification
    classificationLayer('Name', 'output') % Classification output layer
];

options = trainingOptions('sgdm', ... % Correct solver name here
    'MaxEpochs', 10, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 10, ...
    'MiniBatchSize', 8, ...
    'Plots', 'training-progress', ...
    'Verbose', false);




% Train the network
[trainedNet, trainInfo] = trainNetwork(imdsTrain, layers, options);

% Save the trained network
save(outputModelPath, 'trainedNet'); % Save the trained model to the specified path
disp("Trained model saved successfully!");

