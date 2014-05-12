%% visualization for the autoencoder stuff
load('output_test');
for i=1:12
    subplot(3,4,i);
imagesc(reshape(matrix(:,i),28,28));
colormap gray;
end
figure;
load('input_test');
for i=1:12
    subplot(3,4,i);
imagesc(reshape(matrix(:,i),28,28));
colormap gray;
end


load('neural_network.mat');
figure;

for i=130:141
    subplot(3,4,i-129);
imagesc(reshape(W_layer1(i,:)',28,28));
colormap gray;
%caxis([0,1]);

end

%% visualization for the classification stuff
load('output_test');
output=matrix;
load('MNIST/MNIST_test');
[~,decision_classifier]=max(output);
[~,correct]=max(targets);
disp(sum(decision_classifier==correct)/size(targets,2));
coimp=[decision_classifier;correct];
confmat=zeros(10,10);
for i=1:10
    for j=1:10
        confmat(i,j)=sum(decision_classifier==j & correct==i);
    end
end
confmat
