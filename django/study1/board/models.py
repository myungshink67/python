from django.db import models

# Create your models here.
# python manage.py makemigrations
# python manage.py migrate

class Board(models.Model) :
    #num 값이 없으면 자동증가함. (auto increments)
    num = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30)
    pass1 = models.CharField(max_length=20)
    subject = models.CharField(max_length=100)    
    content = models.CharField(max_length=4000)        
    regdate = models.DateTimeField(null=True) #null 허용
    readcnt = models.IntegerField(default=0)
    file1 = models.CharField(max_length=300)
    
    def __str__(self) :
        return str(self.num) + ":" + self.subject
    