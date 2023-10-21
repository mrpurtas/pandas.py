

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")
df.head(10)
#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

cinsiyet_sayısı = df["sex"].value_counts()
cinsiyet_sayısı


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
df[["pclass", "parch"]].nunique()


#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

type(df["embarked"])
df["embarked"] == df["embarked"].astype("category")

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] == "C"]


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] != "S"]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"] < 30) & (df["sex"] == "female")]

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["fare"] > 500) | (df["age"] > 70)]


#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop(columns="who")

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

deck_mode = df["deck"].mode()[0]
df["deck"].fillna(deck_mode, inplace=True)


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
age_medyan = df["age"].median()
df["age"].fillna(age_medyan, inplace=True)
df.head(25)
#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass", "sex"])["survived"].agg(["sum", "count", "sum", "mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
filtered_age = df["age"]
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)
df.head()
#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################


tips = sns.load_dataset("tips")

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips.groupby("time")["total_bill"].agg(["sum", "min", "max", "mean"])

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"])

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################
lunch_female = tips[(tips["time"] == "Lunch") & (tips["sex"] == "Female")]

tips.groupby("day")[["total_bill", "tip"]].agg(["sum", "min", "max", "mean"])
tips.head()

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################
filtered_size = tips[(tips["size"] < 3) & (tips["total_bill"] > 10)]

filtered_size["total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

tips["total_bill_tip_sum"] = tips["total_bill"] + tips["tip"]
tips.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

tips.sort_values(by="total_bill_tip_sum", ascending=False)


def disc(x, y, z):
    disk = y**2 - 4*x*z
    if disk >= 0:
        print("has_real_root")
    else:
        print("has_not_real_root")

        disc(1,2,3)

def to_be_built(a, b):
    if (a / b) >= 2:
        print("YES")
    else:
        print("NO")

        to_be_built(1,2)

def median(scores):
    n = len(scores)
    sorted_scores = sorted(scores)
    if n % 2 == 0:
        medium1 = sorted_scores[n // 2 - 1]
        medium2 = sorted_scores[n // 2]
        median = (medium1 + medium2) // 2
    else:
        median = sorted_scores[n // 2]
    return median

scores1 = [27, 48, 9, 63, 99, 61, 33, 80, 43, 84, 39, 46, 40, 4]

print(median(scores1))

def middle_point(point1, point2):
  x1, y1 = point1
  x2, y2 = point2
  middle_x = (x1 + x2) / 2
  middle_y = (y1 + y2) / 2
    return middle_x, middle_y
#Write a function named sampled_average that takes a list of integers and an integer as a target average. This function return “yes” if we can remove a single element from the given list so that the average of the remaining is equal to target average, prints “no” otherwise.
#
#Sample I/O:
#
#Sample input and output:
#
#>>> sampled_average([1,2,3,10], 2)
#'yes'
#
#>>> sampled_average([1,3], 2)
#'no'
def sample_average(list, x):
    total_sum = sum(list)
    for num in list:
        sum_removal_list = total_sum - num
        new_average = sum_removal_list / (len(list) - 1)
        if new_average == x:
            return "yes"
        else:
            return "no"
print(sample_average([1, 2, 3, 4], 4))


total_point = int(input())

if 0 <= total_point < 50:
    print("FF")
elif 50 <= total_point <= 60:
    print("FD")
elif 60 <= total_point <= 70:
    print("DD")
elif 70 <= total_point <= 80:
    print("CC")
######################################################################################################
Even or Odd 2
In order to decide about the parity of a number (whether it is odd or even) it is enough to inspect its last digit. Write a program that reads a three-digit number from the user. Then checks if the number and reversed form of that number is even or odd (If the number is 123, then the reversed form is 321.) If both are even, print “Even” or if both are odd print “Odd”. If they are different from each other, then print their parity repectively (i.e. if the number is even print “Even Odd” else print “Odd Even”).

Hint: To find whether the reversed version is even or odd, inspecting only the first digit of the actual number is enough.

sayı = int(input())
def tek_cıft(sayı):
    if int(sayı) % 2 == 0:
        if int(sayı / 100) % 2 == 0:
            print("ODD")
        else:
            print("ODD EVEN")
    else:
        if int(sayı / 100) % 2 != 0:
            print("EVEN")
        else:
            print("EVEN ODD")



            tek_cıft(int(input(90)))

######################################################################################################



    for age in ages:
    ages = eval(input())
    total_price = 0
        if 0 <= age <= 10:
            total_price += 30
        elif 11 <= age <= 25:
            total_price += 60
        elif 26 <= age <= 60:
            total_price += 90
        elif 60 < age:
            total_price += 50
    print(total_price)


ages = eval(input())

total_price = 0

for age in ages:
    if 0 <= age <= 10:
        total_price += 30
    elif 11 <= age <= 25:
        total_price += 60
    elif 26 <= age <= 60:
        total_price += 90
    elif 60 < age:
        total_price += 50

print(total_price)

ages = [10, 15, 20]

