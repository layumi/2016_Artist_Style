classdef StyleLoss < dagnn.Loss
    %EPE
    methods
        function outputs = forward(obj, inputs, params)
            [w,h,c,~] = size(inputs{1});
            %0.5*(c-x)^2
            hh = reshape(inputs{1},[],c);
            H = hh'*hh;
            t = bsxfun(@minus,inputs{2},H);
            t = t.^2;
            outputs{1} = sum(sum(t))/(4*w^2*h^2*c^2);
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            [w,h,c,~] = size(inputs{1});
            hh = reshape(inputs{1},[],c);
            H = hh'*hh;
            Y = bsxfun(@minus,H,inputs{2});
            F = reshape(inputs{1},[],c);
            Y = F*Y;
            %Y(F<0) = 0;
            Y = reshape(Y,[w,h,c]);
            Y = Y./(w^2*h^2*c^2);
            derInputs{1} = gpuArray(bsxfun(@times, derOutputs{1},Y));
            derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = StyleLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
