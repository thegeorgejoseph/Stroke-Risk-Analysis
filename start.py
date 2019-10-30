import PySimpleGUI as sg
import pickle
import numpy as np
import time
import sys
# All the stuff inside your window.
layout = [  [sg.Text('Please answer all the questions below to the best of your ability:-')],
            [sg.Text('What is your name?'), sg.InputText()],
            [sg.Text('What is your gender? [male/female/other]'), sg.InputText()],
            [sg.Text('What is your age? [enter with a decimal]'), sg.InputText()],
            [sg.Text('Do you have a history of hypertension? [yes/no]'), sg.InputText()],
            [sg.Text('Do you have a history of heartdisease in your family? [yes/no]'), sg.InputText()],
            [sg.Text('Are you married or unmarried?'), sg.InputText()],
            [sg.Text('What kind of work do you do? [children/private/never-worked/self-employed/government-job]'), sg.InputText()],
            [sg.Text('What is your area of residence?  [urban/rural]'), sg.InputText()],
            [sg.Text('What would you consider your glucose consumption to be like? [0-100/100-200/200-250 (decimals)]'), sg.InputText()],
            [sg.Text('What is your height in metres?'), sg.InputText()],
            [sg.Text('What is your body weight in kilograms?'), sg.InputText()],
            [sg.Text('What is your activity in terms of smoking? [never-smoked/formerly-smoked/smokes]'), sg.InputText()],
            [sg.Submit(), sg.Button('Cancel')] ]

# Create the Window
window = sg.Window('Stroke Risk Analysis with Intelligent Nutrition System', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Cancel'):	# if user closes window or clicks cancel
        break
    print('Thank You for your response!', values[0])

window.close()

name = values[0]

if values[1]=='male':
    gender = 1
else:
    gender = 0

age = float(values[2])

if values[3]=='yes':
    hypertension = 1
else:
    hypertension = 0

if values[4]=='yes':
    heartdisease = 1
else:
    heartdisease = 0

if values[5]=="married":
    ever_married = 1
else:
    ever_married = 0

if values[6]=="children":
    work_type = 0
elif values[6]=="private":
    work_type = 1
elif values[6]=="never-worked":
    work_type = 2
elif values[6]=="self-employed":
    work_type = 3
elif values[6]=="government-job":
    work_type = 4

if values[7]=="urban":
    residence_type = 1
else:
    residence_type = 0

glucose = float(values[8])

kilograms = float(values[10])
meters = float(values[9])

bmi = kilograms/(meters*meters)

if values[11]=="never-smoked":
    smoke=0
elif values[11]=="formerly-smoked":
    smoke=1
elif values[11]=="smokes":
    smoke=2

array = np.array([(gender,age,hypertension,heartdisease,ever_married,work_type,residence_type,glucose,bmi,smoke),(0,45.0,1,0,1,1,0,89.82,28.4,1)])
print()
print()
print()
#print(array)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
new_predictions = loaded_model.predict_proba(array)[:,1]
result = new_predictions[0]
print("The probability of you getting a stroke is :--",result)


if result>0.0 and result<=0.35:
    risk="low"
if result>0.35 and result<=0.60:
    risk="medium"
if result >0.60:
    risk="high"

hyperlifestyle = ["Lose extra pounds and watch your waistline.Men are at risk if their waist measurement is greater than 40 inches (102 centimeters).Women are at risk if their waist measurement is greater than 35 inches (89 centimeters).","Exercise regularly","Eat a healthy diet.Keep a food diary. Writing down what you eat, even for just a week, can shed surprising light on your true eating habits. Monitor what you eat, how much, when and why.Consider boosting potassium. Potassium can lessen the effects of sodium on blood pressure. The best source of potassium is food, such as fruits and vegetables, rather than supplements. Talk to your doctor about the potassium level that's best for you.Be a smart shopper. Read food labels when you shop and stick to your healthy-eating plan when you are dining out, too.","Reduce sodium in your diet","Limit the amount of alcohol you drink","Quit smoking","Cut back on caffeine","Reduce your stress","Monitor your blood pressure at home and see your doc"]

heartlifestyle = ["Choose foods low in saturated fat, trans fat, and sodium."," As part of a healthy diet, eat plenty of fruits and vegetables, fiber-rich whole grains, fish (preferably oily fish-at least twice per week), nuts, legumes and seeds and try eating some meals without meat.","Select lower fat dairy products and poultry (skinless)."," Limit sugar-sweetened beverages and red meat. If you choose to eat meat, select the leanest cuts available.","Be physically active.","learn the warning signs of a heart attack and stroke."]

sg.Popup("YOU HAVE ", risk.upper() ,"PROBABILITY OF RISK OF STROKE")
sg.Popup("You should consider the following lifestyle changes!")
if hypertension ==1:
    for i in hyperlifestyle:
        sg.Popup(i)
        #time.sleep(1)
if heartdisease ==1:
    for i in heartlifestyle:
        sg.Popup(i)
        #time.sleep(1)

print("Thank you, ",name,", for using this application!")
time.sleep(5)
sys.exit()
