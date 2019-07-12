import csv
from code import Code
from DPT import *
from tqdm import tqdm
from numba import jit
# import cv2


train_path = '/data2/lt/ctr/train/csv/train_20190518.csv'
test_path = '/data2/lt/ctr/train/csv/test_20190518.csv'
user_path = '/data2/lt/ctr/train/csv/user_info.csv'
ad_path = '/data2/lt/ctr/train/csv/ad_info.csv'
content_path = '/data2/lt/ctr/train/csv/content_info.csv'
code_train_path = '/data2/lt/ctr/train/npy/'
code_test_path = '/data2/lt/ctr/train/npy/'

# time1 = cv2.getTickCount()
user_dict = {}
user_f = open(user_path, 'r')
user_reader = csv.reader(user_f)
print('user')
for line in tqdm(user_reader):
    uId, age, gender, city, privince, phoneType, carrier = line
    if age == 'NULL':
        age = '7'
    if gender == 'NULL':
        gender = '2'
    user_dict[uId] = (age, gender)
user_f.close()
# time1 = cv2.getTickCount() - time1
# time1 = time1 / cv2.getTickFrequency()


# time2 = cv2.getTickCount()
print('ad')
ad_dict = {}
ad_f = open(ad_path, 'r')
ad_reader = csv.reader(ad_f)
for line in tqdm(ad_reader):
    adId, billId, primId, creativeType, interType, spreadAppId = line
    ad_dict[adId] = (billId, creativeType, interType)
ad_f.close()
# time2 = cv2.getTickCount() - time2
# time2 = time2 / cv2.getTickFrequency()


# time3 = cv2.getTickCount()
print('content')
content_dict = {}
content_f = open(content_path, 'r', encoding='utf-8')
content_reader = csv.reader(content_f)
for line in tqdm(content_reader):
    contentId, firstClass, secondClass = line
    content_dict[contentId] = firstClass
content_f.close()
# time3 = cv2.getTickCount() - time3
# time3 = time3 / cv2.getTickFrequency()


# time4 = cv2.getTickCount()
train_samples = []
train_f = open(train_path, 'r')
train_reader = csv.reader(train_f)
code = Code()
print('train is coding...')
@jit
def traincode():
    for line in tqdm(train_reader):
        label, uId, adId, operTime, siteId, slotId, contentId, netType = line
        if not user_dict.__contains__(uId):
            age, gender = code.zeros_age(), code.zeros_gender()
        else:
            age, gender = user_dict[uId]
        if not ad_dict.__contains__(adId):
            billId, creativeType, interType = code.zeros_bill(), code.zeros_creative(), code.zeros_inter()
        else:
            billId, creativeType, interType = ad_dict[adId]
        if not content_dict.__contains__(contentId):
            firstClass = -1
        else:
            firstClass = content_dict[contentId]
        sample = code.code(label, siteId, netType, age, gender, billId, creativeType, interType, firstClass)
        train_samples.append(sample)
    train_f.close()
    IO.writeData(code_train_path, np.array(train_samples, dtype=bool), 'code_train')
traincode(3)
print('test is coding...')
test_samples = []
test_f = open(test_path, 'r')
test_reader = csv.reader(test_f)

for line in tqdm(test_reader):
    label, uId, adId, operTime, siteId, slotId, contentId, netType = line
    age, gender = user_dict[uId]
    billId, creativeType, interType = ad_dict[adId]
    if not content_dict.__contains__(contentId):
        firstClass = -1
    else:
        firstClass = content_dict[contentId]
    # firstClass = content_dict[contentId]
    sample = code.code(label, siteId, netType, age, gender, billId, creativeType, interType, firstClass)
    test_samples.append(sample)
test_f.close()
IO.writeData(code_test_path, np.array(test_samples, dtype=bool), 'code_test')




