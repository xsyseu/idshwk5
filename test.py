from sklearn.ensemble import RandomForestClassifier
import math


def num_in_str(s):
    count = 0
    for c in s:
        if c.isdigit():
            count += 1
    return count


def cal_entropy(s):
    h = 0.0
    sumt = 0
    letter = [0] * 26
    s = s.lower()
    for c in s:
        if c.isalpha():
            letter[ord(c) - ord("a")] += 1
            sumt += 1
    for i in range(26):
        p = 1.0 * letter[i] / sumt
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


def cal_segmentation(s):
    count = 0
    for c in s:
        if c == ".":
            count += 1
    return count


def initData(filename, domainlist):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            domain_name = tokens[0]
            if len(tokens) == 1:
                label = "unknown"
            else:
                label = tokens[1]
            length = len(domain_name)
            num = num_in_str(domain_name)
            entropy = cal_entropy(domain_name)
            seg_num = cal_segmentation(domain_name)
            domainlist.append(Domain(domain_name, length, num, entropy, seg_num, label))


class Domain:
    def __init__(self, _name, _length, _num, _entropy, _seg_num, _label):
        self.name = _name
        self.length = _length
        self.num = _num
        self.entropy = _entropy
        self.seg_num = _seg_num
        self.label = _label

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

    def returnData(self):
        return [self.length, self.num, self.entropy, self.seg_num]


def main():
    train_domainlist = []
    test_domainlist = []
    initData("train.txt", train_domainlist)
    initData("test.txt", test_domainlist)
    featureMatrix = []
    labelList = []
    for item in train_domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    with open("result.txt", "w") as f:
        for item in test_domainlist:
            f.write(item.name)
            f.write(",")
            if clf.predict([item.returnData()])[0] == 0:
                f.write("notdga")
            else:
                f.write("dga")
            f.write("\n")


if __name__ == '__main__':
    main()
