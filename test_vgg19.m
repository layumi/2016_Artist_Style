clear;
%----data
im1 = single(imresize(imread('1.jpg'),[400,512]));
im2 = single(imresize(imread('4.jpg'),[400,512]));
input = rand(400,512,3,'single');
im1 = bsxfun(@minus,im1,reshape([123.6800,116.7790,103.9390],[1,1,3]));
im2 = bsxfun(@minus,im2,reshape([123.6800,116.7790,103.9390],[1,1,3]));
%----net initialization
net = load('./data/imagenet-vgg-verydeep-19.mat');
net = vl_simplenn_tidy(net) ;
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
%net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
%{'prediction','label'}, 'error') ;
net.mode = 'test' ;
net.conserveMemory = false;
net.move('gpu');
for i=1:numel(net.params)
    net.params(i).learningRate = 0;
end

%----get content label
net.eval({'input',gpuArray(im2)});
i = net.getVarIndex('x22'); %conv4_2
labelc = net.vars(i).value;

%----content loss
net.addLayer('loss_c',ContentLoss(),{'x22','labelc'},'objective_c');


%----get style label
net.eval({'input',gpuArray(im1)});
i1 = net.getVarIndex('x1'); %conv1_1
i2 = net.getVarIndex('x6'); %conv2_1
i3 = net.getVarIndex('x11'); %conv3_1
i4 = net.getVarIndex('x20'); %conv4_1
i5 = net.getVarIndex('x29'); %conv5_1
labels_1 = net.vars(i1).value;
N1 = size(labels_1,3);
labels_1 = reshape(labels_1,[],N1);
H1 = labels_1'*labels_1;

labels_2 = net.vars(i2).value;
N2 = size(labels_2,3);
labels_2 = reshape(labels_2,[],N2);
H2 = labels_2'*labels_2;

labels_3 = net.vars(i3).value;
N3 = size(labels_3,3);
labels_3 = reshape(labels_3,[],N3);
H3 = labels_3'*labels_3;

labels_4 = net.vars(i4).value;
N4 = size(labels_4,3);
labels_4 = reshape(labels_4,[],N4);
H4 = labels_4'*labels_4;

labels_5 = net.vars(i5).value;
N5 = size(labels_5,3);
labels_5 = reshape(labels_5,[],N5);
H5 = labels_5'*labels_5;

%----style loss
net.addLayer('loss_s1',StyleLoss(),{'x1','labels_1'},'objective_s1'); 
net.addLayer('loss_s2',StyleLoss(),{'x6','labels_2'},'objective_s2'); 
net.addLayer('loss_s3',StyleLoss(),{'x11','labels_3'},'objective_s3'); 
net.addLayer('loss_s4',StyleLoss(),{'x20','labels_4'},'objective_s4'); 
net.addLayer('loss_s5',StyleLoss(),{'x29','labels_5'},'objective_s5'); 


%----train
net.mode = 'normal' ;
net.accumulateParamDers = true;
derOutputs = {'objective_c',0.2*1e-3,'objective_s1',0.2,'objective_s2',0.2,...
    'objective_s3',0.2,'objective_s4',0.2,'objective_s5',0.2};
opts.gamma = 0.99;
opts.constraint = 100;
opts.momentum = 0.9;
momentum = gpuArray(zeros(size(input),'single'));
r = gpuArray(zeros(size(input),'single'));
for i=1:6000
    net.eval({'input',gpuArray(input),'labelc',labelc,'labels_1',H1,...
        'labels_2',H2,'labels_3',H3,'labels_4',H4,'labels_5',H5}, derOutputs) ;
    r = (1 - opts.gamma) * (net.vars(1).der).^2 + opts.gamma * r;
    der_constraint = net.vars(1).der;
    der_constraint(der_constraint>opts.constraint) = opts.constraint;
    der_constraint(der_constraint<-opts.constraint) = -opts.constraint;
    momentum = opts.momentum * momentum ...
        -  der_constraint ./ (sqrt(r)+1e-9);  
    input = input + 0.1* momentum; %update
    fprintf('epoch:%d object_c:%f  objective_s3:%f  \n',...
        i,net.vars(46).value,net.vars(52).value);
    if(mod(i,1000)==0)
        re_im = uint8(bsxfun(@plus,gather(input),reshape([123.6800,116.7790,103.9390],[1,1,3])));
        imshow(re_im);
    end
end
imwrite(re_im,'demo.jpg');