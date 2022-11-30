from django.http import  HttpResponse
from django.shortcuts import render
from django.contrib import  auth
import pyrebase


# arr = [2,"four ", "six", 8]
# def hello_world(request):
#
#     return  HttpResponse("Hello World we got this! " )


config = {

    'apiKey': "AIzaSyAWuD0RF1NXotnqieNh-FkFhf730lQ0PAc",

    'authDomain': "sd3-group-project.firebaseapp.com",

    'projectId': "sd3-group-project",

    'storageBucket': "sd3-group-project.appspot.com",

    'messagingSenderId': "918599092511",

    'appId': "1:918599092511:web:93712ecd7113de0c56c82a",

    'databaseURL': "https://sd3-group-project-default-rtdb.europe-west1.firebasedatabase.app/"

}
firebase = pyrebase.initialize_app(config)

authent = firebase.auth()

database = firebase.database()


def signIn(request):
    return render(request,"signIn.html")



def postsign(request):
    email = request.POST.get('email')
    passw = request.POST.get("pass")
    print(email,passw)

    try:
        user = authent.sign_in_with_email_and_password(email,passw)
    except:
        message="Invalid Credentials"
        return render(request,"signIn.html",{"m":message})

    print(user['idToken'])
    # Creating a session token
    session_id = user['idToken']
    request.session['uid'] = str(session_id)

    return render(request,"welcome.html",{"e":email})

def postregister(request):
    name = request.POST.get('name')
    email = request.POST.get('email')
    passw = request.POST.get("pass")
    print(email,passw)

    # Creating Account

    try:
         user = authent.create_user_with_email_and_password(email,passw)
    # Getting Unique User id
    except:
        message = "Account could not be created try again!"
        return render(request,"register.html",{"m":message})

    uid = user['localId']

    data = {"name":name,"status":"1"}

    database.child("users").child(uid).child("details").set(data)

    return render(request,"signIn.html")

def home(request):
    return render(request,"home.html")


def logout(request):
    auth.logout(request)
    return render(request,'signIn.html')

def register(request):
    return render(request,"register.html")




