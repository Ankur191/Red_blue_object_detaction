import math
class box:
    def get_area(self,box):
        x1, y1, x2, y2=box
        w=abs(x2-x1)
        h=abs(y2-y1)
        return w*h
    def get_intersection_points(self,box1,box2):
        x1,y1,x2,y2=box1

        x3,y3,x4,y4=box2

        x_inter1=max(x1,x3)
        y_inter1=max(y1,y3)

        x_inter2=min(x2,x4)
        y_inter2=min(y2,y4)

        return x_inter1,y_inter1,x_inter2,y_inter2

    def get_IOU(self,inter,union):
        return inter/union

    def get_center(self,box):
        x1,y1,x2,y2=box
        x=abs(x2-x1)/2
        y=abs(y2-y1)/2
        return x,y

    def get_diagonal_length(self,box1,box2):
        x1,y1,x2,y2=box1
        x3,y3,x4,y4=box2

        d_x1=min(x1,x2,x3,x4)
        d_y1=min(y1,y2,y3,y4)
        d_x2 = max(x1, x2, x3, x4)
        d_y2 = max(y1, y2, y3, y4)
        return math.sqrt((d_x1-d_x2)**2+(d_y1-d_y2)**2)

    def get_center_dist(self,c1,c2):
        x1,y1=c1
        x2,y2=c2
        ans=math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return ans
    def get_distance_IOU(self,iou,box1,box2):
        c1=self.get_center(box1)
        c2=self.get_center(box2)
        d=self.get_center_dist(c1,c2)
        c=self.get_diagonal_length(box1,box2)
        return (1-iou+((d**2)/(c**2)))

box1=[4,4,2,2]
box2=[4,4,2,8]

b1=box()
x1,y1,x2,y2=b1.get_intersection_points(box1,box2)

inter_box=[x1,y1,x2,y2]
area_box1=b1.get_area(box1)
area_box2=b1.get_area(box2)

inter_area=b1.get_area(inter_box)
union_area=area_box1+area_box2-inter_area

#IOU
iou=b1.get_IOU(inter_area,union_area)

# distance IOU(1-iou+d^2/c^2)
dist_iou=b1.get_distance_IOU(iou,box1,box2)
print(dist_iou)

#complete iou(ciou) remaining

