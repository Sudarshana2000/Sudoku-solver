from django.db import models

# Create your models here.
class Image(models.Model):
    photo=models.ImageField()
    date=models.DateTimeField(auto_now_add=True)
    solved=models.ImageField(default=None)