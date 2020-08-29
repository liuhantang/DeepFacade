clc,close    
BW=imread('monge_2_mask.png');  
 for i=1:size(BW,1)
        for j=1:size(BW,2)
            if (BW(i,j,1)==255) && (BW(i,j,2)==0) && (BW(i,j,3)==0)
%                 BW(i,j)=1;%Window 
            else BW(i,j,:) = [128,255,255];
            end
            
        end
 end

figure(1);imshow(BW,[]);title('red')

se1=strel('disk',5);%这里是创建一个半径为5的平坦型圆盘结构元素  

BW=rgb2gray(BW);    
figure(3),imshow(BW,[]);


se1=strel('disk',1);%这里是创建一个半径为5的平坦型圆盘结构元素  
A2=imdilate(BW,se1); 
figure(2);
imshow(A2);
title('erode');

BW = A2
thresh=[0.01,0.17];    
sigma=2;%定义高斯参数    
f = edge(double(BW),'canny',thresh,sigma);    
figure(4),imshow(f,[]);    
title('canny 边缘检测');    
    
[H, theta, rho]= hough(f,'RhoResolution', 0.5);    
%imshow(theta,rho,H,[],'notruesize'),axis on,axis normal    
%xlabel('\theta'),ylabel('rho');    
    
peak=houghpeaks(H,5);    
hold on    
    
lines=houghlines(f,theta,rho,peak);    
figure,imshow(f,[]),title('Hough Transform Detect Result'),hold on    
for k=1:length(lines)    
    xy=[lines(k).point1;lines(k).point2];    
    plot(xy(:,1),xy(:,2),'LineWidth',4,'Color',[.6 .6 .6]);    
end    









