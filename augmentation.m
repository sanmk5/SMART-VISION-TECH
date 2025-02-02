
inputFolder = "D:\FLIPKART EVENT\DATASET"; 
outputFolder = "D:\FLIPKART EVENT"; 


if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

% Create an imageDatastore to load images from the input folder
imds = imageDatastore(inputFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Augmentation and processing loop
numAugmentedImagesPerImage = 10; % Number of augmented images per original image
imageSize = [224, 224]; % Resize to this size (e.g., for ResNet50 input)

% Loop through each image in the datastore
while hasdata(imds)
    % Read the image and its label
    [img, info] = read(imds);
    
    % Ensure the image is in the correct format (RGB)
    if size(img, 3) ~= 3
        img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB
    end
    
    % Resize the original image
    img = imresize(img, imageSize);
    
    % Generate augmented images
    for i = 1:numAugmentedImagesPerImage
        % Apply random transformations
        augImg = img;
        
        % Random rotation
        angle = randi([-20, 20]); % Random angle between -20 and 20 degrees
        augImg = imrotate(augImg, angle, 'crop');
        
        % Random translation
        tx = randi([-10, 10]); % Horizontal shift
        ty = randi([-10, 10]); % Vertical shift
        augImg = imtranslate(augImg, [tx, ty]);
        
        % Random horizontal flip
        if rand > 0.5
            augImg = flip(augImg, 2);
        end
        
        % Random brightness adjustment
        brightnessFactor = 0.8 + 0.4 * rand; % Scale between 0.8 and 1.2
        augImg = imadjust(augImg, [], [], brightnessFactor);
        
        % Random contrast adjustment
        contrastFactor = 0.8 + 0.4 * rand; % Scale between 0.8 and 1.2
        augImg = imadjust(augImg, stretchlim(augImg), [], contrastFactor);
        
        % Create the filename for the augmented image
        [~, name, ext] = fileparts(info.Filename);
        augmentedFilename = fullfile(outputFolder, ...
            sprintf('%s_aug%d%s', name, i, ext));
        
        % Save the augmented image
        imwrite(augImg, augmentedFilename);
    end
end

disp('Data augmentation completed! Augmented images saved to the output folder.');
