from django.shortcuts import render
from .forms import ImageForm
from .models import Image
from .sudoku_solver import process
import cv2
from django.conf import settings
from django.core.files import File

# Create your views here.

def home(request):
    form=ImageForm()
    img=None
    flag=False
    if request.method=="POST":
        form=ImageForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
        img=Image.objects.last()
    context={'form':form, 'img':img}
    return render(request,'home.html',context)

def solveSudoku(request):
    if request.method=="POST":
        img=Image.objects.last()
        source=cv2.imread(img.photo.path)
        output=process(source)
        path=settings.MEDIA_ROOT+"/solved_"+img.photo.name
        cv2.imwrite(path,output)
        img.solved.save("solved_"+img.photo.name,File(open(path,'rb')),save=True)
    context={'img':img}
    return render(request,'home.html',context)