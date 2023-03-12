from django.db import models

#class Point(models.Model):

    #def __str__(self):
        #return [{self.x1, self.y1}, {self.x2, self.y2}, { self.x3, self.y3}, {self.x4, self.y4} ]
        #return f"({self.x1} {self.y1} {self.x2} {self.y2} {self.x3} {self.y3} {self.x4} {self.y4}"

class LoopInfo(models.Model):
    name = models.CharField(max_length=50, default="loop1")
    loop_id = models.IntegerField(default=0)
    x1 = models.IntegerField(default=900)
    y1 = models.IntegerField(default=600)
    x2 = models.IntegerField(default=900)
    y2 = models.IntegerField(default=300)
    x3 = models.IntegerField(default=400)
    y3 = models.IntegerField(default=300)
    x4 = models.IntegerField(default=400)
    y4 = models.IntegerField(default=600)
    #points = models.ForeignKey(Point, on_delete=models.CASCADE)
        #x1=900,y1=600,x2=900,y2=200,x3=400,y3=300,x4=400,y4=600))
    orientation = models.CharField(max_length=50)
    x = models.IntegerField(default=20)
    y = models.CharField(max_length=50,default="20")
    #summary_location = models.JSONField(Point, on_delete=models.CASCADE, related_name='loop_summary_location')

    def __str__(self):
        return f"{self.name}"