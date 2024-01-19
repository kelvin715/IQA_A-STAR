for i = 1:length(children_folders)
    % 跳过 '.' 和 '..' 文件夹
    if strcmp(children_folders(i).name, '.') || strcmp(children_folders(i).name, '..')
        continue;
    end
    
    % 训练
    train_dir = fullfile(parent_folder, children_folders(i).name, 'train', 'images');
    imds = imageDatastore(train_dir, 'FileExtensions', {'.jpg'});
    model = fitniqe(imds);

    % 测试
    folder = fullfile(parent_folder, children_folders(i).name, 'test', 'images'); 
    images = dir(fullfile(folder, '*.jpg'));  % 修正了这里的引号
    niqe_scores = zeros(length(images), 1);

    for j = 1:length(images)
        image_path = fullfile(folder, images(j).name);
        img = imread(image_path);
        niqe_scores(j) = niqe(img, model);
    end

    average_niqe = mean(niqe_scores);  % 计算平均得分
    disp(['Average NIQE score for ', children_folders(i).name, ': ', num2str(average_niqe)]);  % 显示平均得分
end