% Load the trained model
load("D:\FLIPKART EVENT\trainedmodel1.mat"); % Replace 'trained_model_brandname.mat' with your model's filename

% Provide the path to the input image
imgPath ="D:\FLIPKART EVENT\RAW IMAGES\frontside.png"; % Replace with the actual path of the input image
inputImage = imread(imgPath);

% Resize the image to match the input size of the trained model
inputSize = trainedNet1.Layers(1).InputSize(1:2); % Get model's input size (e.g., [224, 224])
imgResized = imresize(inputImage, inputSize);

% Normalize the image for prediction
imgNormalized = im2single(imgResized); % Convert image to single precision

% Predict the brand name
[predictedLabel, scores] = classify(trainedNet1, imgNormalized);

% Display the result
disp(['Predicted Brand Name: ', char(predictedLabel)]);
disp(['Confidence Score: ', num2str(max(scores))]);

% Show the image with predicted brand name
figure;
imshow(inputImage);
title(['Predicted: ', char(predictedLabel), ' (Confidence: ', num2str(max(scores)), ')']);
