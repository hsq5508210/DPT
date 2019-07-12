import numpy as np


class Code:
    def __init__(self):
        self.site_num = 12
        self.net_num = 6
        self.age_num = 7
        self.gender_num = 3
        self.bill_num = 2
        self.creative_num = 5
        self.inter_num = 4
        self.firstClass_num = 23
        self.site_dict = {}
        self.net_dict = {}
        self.age_dict = {}
        self.gender_dict = {}
        self.bill_dict = {}
        self.creative_dict = {}
        self.inter_dict = {}
        self.firstClass_dict = {}

    def code(self, label, siteId, netType, age, gender, billId, creativeType, interType, firstClass):
        label_code = np.array([label])
        site_code = self.get_site_code(siteId)
        net_code = self.get_net_code(netType)
        age_code = self.get_age_code(age)
        gender_code = self.get_gender_code(gender)
        bill_code = self.get_bill_code(billId)
        creative_code = self.get_creative_code(creativeType)
        inter_code = self.get_inter_code(interType)
        firstClass_code = self.get_firstClass_code(firstClass)
        sample = np.concatenate((label_code, site_code, net_code, age_code, gender_code, bill_code, creative_code, inter_code,
                           firstClass_code)).astype(np.bool)
        return sample

    def zeros_age(self):
        return np.zeros(self.age_num).astype(np.int)

    def zeros_gender(self):
        return np.zeros(self.gender_num).astype(np.int)

    def zeros_bill(self):
        return np.zeros(self.bill_num).astype(np.int)

    def zeros_creative(self):
        return np.zeros(self.creative_num).astype(np.int)

    def zeros_inter(self):
        return np.zeros(self.inter_num).astype(np.int)

    def zeros_firstClass(self):
        return np.zeros(self.firstClass_num).astype(np.int)

    def get_site_code(self, siteId):
        if self.site_dict.__contains__(siteId):
            site_code = self.site_dict[siteId]
        else:
            length = self.site_dict.__len__()
            index = length
            value = np.zeros(self.site_num, np.int)
            value[index] = 1
            self.site_dict[siteId] = value
            site_code = value
        return site_code

    def get_net_code(self, netType):
        if self.net_dict.__contains__(netType):
            net_value = self.net_dict[netType]
        else:
            length = self.net_dict.__len__()
            index = length
            value = np.zeros(self.net_num, np.int)
            value[index] = 1
            self.net_dict[netType] = value
            net_value = value
        return net_value

    def get_age_code(self, age):
        if self.age_dict.__contains__(age):
            age_value = self.age_dict[age]
        else:
            length = self.age_dict.__len__()
            index = length
            value = np.zeros(self.age_num, np.int)
            value[index] = 1
            self.age_dict[age] = value
            age_value = value
        return age_value

    def get_gender_code(self, gender):
        if self.gender_dict.__contains__(gender):
            gender_value = self.gender_dict[gender]
        else:
            length = self.gender_dict.__len__()
            index = length
            value = np.zeros(self.gender_num, np.int)
            value[index] = 1
            self.gender_dict[gender] = value
            gender_value = value
        return gender_value

    def get_bill_code(self, billId):
        if self.bill_dict.__contains__(billId):
            bill_value = self.bill_dict[billId]
        else:
            length = self.bill_dict.__len__()
            index = length
            value = np.zeros(self.bill_num, np.int)
            value[index] = 1
            self.bill_dict[billId] = value
            bill_value = value
        return bill_value

    def get_creative_code(self, creativeType):
        if self.creative_dict.__contains__(creativeType):
            creative_value = self.creative_dict[creativeType]
        else:
            length = self.creative_dict.__len__()
            index = length
            value = np.zeros(self.creative_num, np.int)
            value[index] = 1
            self.creative_dict[creativeType] = value
            creative_value = value
        return creative_value

    def get_inter_code(self, interType):
        if self.inter_dict.__contains__(interType):
            inter_value = self.inter_dict[interType]
        else:
            length = self.inter_dict.__len__()
            index = length
            value = np.zeros(self.inter_num, np.int)
            value[index] = 1
            self.inter_dict[interType] = value
            inter_value = value
        return inter_value

    def get_firstClass_code(self, firstClass):
        if firstClass == -1:
            firstClass_value = self.zeros_firstClass()
        elif self.firstClass_dict.__contains__(firstClass):
            firstClass_value = self.firstClass_dict[firstClass]
        else:
            length = self.firstClass_dict.__len__()
            index = length
            value = np.zeros(self.firstClass_num, np.int)
            value[index] = 1
            self.firstClass_dict[firstClass] = value
            firstClass_value = value
        return firstClass_value
