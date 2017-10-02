#include "math.h"
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>
#include <string>
#include <cstring>
#include <map>

using namespace std;

vector<int> genVec(int vectorLen);

void desVec(vector<int> &res);

void returnMinMax(vector<int> vec, int &aMin, int &aMax);

void testReturnMinMax();

void quickSort(vector<int> &vec, int i, int j);

void testQuickSort();

void knapSack(vector<int> vec, vector<int> &vFirst, vector<int> &vSecond);  

void testKnapSack();

void counterLen(string str, int &strLen);

void testCounterLen();

void testHeap();

void testVector();

void maxSubVec(vector<double> vec, double &maxSub);

void testMaxSubVec();

void num2String(int num, string &str);

void testNum2String();

void testTran();

void testMicrosoft();

string hexAdd(string str1, string str2);

string hexMul(string str1, string str2);

void testNetEast();

void testMomenta();

int main() {
    cout << "Hello, Word!" << endl;
    /*
    vector<int> res = genVec(5);
    desVec(res);
    */
    // testReturnMinMax();
    // testQuickSort();
    // testKnapSack();
    // testCounterLen();
    // testHeap();
    // testVector();
    // testMaxSubVec();
    // testNum2String();
    // testTran();
    // testMicrosoft();
    // string res = hexMul("1278979879", "1269962892749274265927");
    // cout << res << endl;
    // testNetEast();
    testMomenta();

}

vector<int> genVec(int vectorLen) {
    vector<int> result;
    while (vectorLen) {
        result.push_back(vectorLen--);
    }
    return result;
}

void desVec(vector<int> &res) {
    while (!res.empty()) {
        cout << res.back() << endl;
        res.pop_back();
    }
}

void returnMinMax(vector<int> vec, int &aMin, int &aMax) {
    aMin = vec[0];
    aMax = vec[0];
    for (int i = 1; i < vec.size(); ++i) {
        if (vec[i] > aMax)
            aMax = vec[i];
        else if (vec[i] < aMin)
            aMin = vec[i];
    }
}

void testReturnMinMax() {
    int a[] = {1, 2, 4, 90, -3};
    vector<int> myVec(a, a + 5);
    int vMin, vMax;
    returnMinMax(myVec, vMin, vMax);
    cout << "vmin:" << vMin << "  vMax:" << vMax << endl;
}

void quickSort(vector<int> &vec, int i, int j) {
    if (i < j) {
        int temp = vec[i];
        int left = i;
        int right = j;

        while (left < right) {
            while (vec[right] > temp && left < right) --right;
            if (left < right) {
                vec[left] = vec[right];
                ++left;
            }
        
            while (vec[left] < temp && left < right) ++left;
            if (left < right) {
                vec[right] = vec[left];
                --right;
            }
        }

        vec[right] = temp;

        quickSort(vec, i, right - 1);
        quickSort(vec, right + 1, j);
    }
}


void testQuickSort() {
    int a[] = {8, 4, 7, 2, 4};
    vector<int> myVec(a, a + 5);
    cout << "vector is finished" << endl;
    quickSort(myVec, 0, myVec.size() - 1);
    
    vector<int>::iterator iter;
    for (iter = myVec.begin(); iter != myVec.end(); ++iter) {
        cout << *iter << endl;
    }

}

void knapSack(vector<int> vec, vector<int> &vFirst, vector<int> &vSecond) {
    int vecLen = vec.size();
    int vecSum = 0;
    
    for (int i = 0; i < vec.size(); ++i) vecSum += vec[i];
    vecSum = (int) (vecSum / 2);
    
    // store the final result
    vector<vector<int> > res(vecLen + 1, vector<int>(vecSum + 1));

    // dynamic process
    for (int i = 1; i <= vec.size(); ++i) {
        for (int j = 1; j <= vecSum; ++j) {
            if (j < vec[i - 1])
                res[i][j] = res[i - 1][j];
            else
                res[i][j] = max(res[i - 1][j], res[i - 1][j - vec[i - 1]] + vec[i - 1]);
        }
    }

    for (int i = vec.size(); i > 0; --i) {
        if (res[i][vecSum] > res[i - 1][vecSum]) {
            vFirst.push_back(vec[i]);
            vecSum -= vec[i - 1];
        } else {
            vSecond.push_back(vec[i]);
        }
    }

}


void testKnapSack() {
    int a[] = {2, 4, 5, 6, 7};
    vector<int> vec(a, a + 5);
    vector<int> first, second;

    // test the function
    knapSack(vec, first, second);

    for (vector<int>::iterator iter = first.begin(); iter != first.end(); ++iter) {
        cout << *iter << endl;
    }
}


void counterLen(string str, int &strLen) {
    int temp = 0;
    stack<int> myStack;

    for (int i = 0; i < str.size(); ++i) {
        if (str[i] == ')') {
            if (myStack.empty()) temp = 0;
            else {
                myStack.pop();
                temp += 2;
                strLen = max(temp, strLen);
            }
        }
        else myStack.push(1);
    }
}


void testCounterLen() {
    string str = "(()";
    int maxLen = 0;
    counterLen(str, maxLen);
    cout << maxLen << endl;
}



void showVector(vector<int> vec) {
    for (int i = 0; i < vec.size(); ++i) cout << vec[i] << " ";
    cout << endl;
}


void testHeap() {
    int a[] = {0, 3, 5, 7};
    int aLen = (int)(sizeof(a) / sizeof(a[0]));
    vector<int> myVec(a, a + aLen);

    make_heap(myVec.begin(), myVec.end());
    showVector(myVec);

    myVec.push_back(17);
    push_heap(myVec.begin(), myVec.end());
    showVector(myVec);

    pop_heap(myVec.begin(), myVec.end());
    showVector(myVec);

    cout << myVec.back() << endl;

    myVec.pop_back();
    showVector(myVec);

    sort_heap(myVec.begin(), myVec.end());
    showVector(myVec);

}


void testVector() {
    int a[] = {3, 4, 6, 78, 23, 76};
    vector<int> myVec(a, a + (int)(sizeof(a)/sizeof(a[0])));
    
    cout << "Print the vector" << endl;
    for (int i = 0; i < myVec.size(); ++i) {
        cout << myVec[i] << " ";
    }
    cout << endl;

    cout << "Sort the vector" << endl;
    sort(myVec.begin(), myVec.end());
    for (vector<int>::iterator iter = myVec.begin(); iter != myVec.end(); ++iter) {
        cout << *iter << " ";
    }
    cout << endl;
}


void maxSubVec(vector<double> vec, double &maxSub) {
    double maxVec = 1;
    double minVec = 1;

    for (int i = 0; i < vec.size(); ++i) {
        maxVec = max(maxVec * vec[i], max(vec[i], minVec * vec[i]));
        minVec = min(minVec * vec[i], min(vec[i], maxVec * vec[i]));

        maxSub = max(maxSub, maxVec);
    }

}

void testMaxSubVec() {
    double a[] = {-0.5, 0.7, 89, 43, 0.6, -0.9};
    vector<double> myVec(a, a + (int)(sizeof(a)/sizeof(a[0])));

    double result = 0.0;
    maxSubVec(myVec, result);
    
    cout << result << endl;
}


void num2String(int num, string &str) {
    stringstream stream;
    stream << num;
    str = stream.str();
}


void string2Num(int &num, const string str) {
    stringstream stream(str);
    stream >> num;
}


void testNum2String() {
    int a = 12390;
    string res = "";
    num2String(a, res);
    cout << res << endl;
}


void testTran() {
    int num = 238974;
    string str = to_string(num);
    cout << str << endl;

    string s = "234";
    double d = stod(s);
    int ss = stoi(s);
    cout << "int num: " << ss << "double num: " << d << endl;

    for (auto s : str) cout << s << " ";
    cout << endl;

    char a[] = {'H', 'e', 'K', 'H'};
    char aa[sizeof(a)/sizeof(*a)];
    auto ret = copy(begin(a), end(a), aa);

    vector<char> vecChar;
    for (auto s : aa) {
        cout << s << " ";
        vecChar.push_back(s);
    }

    cout << endl;
    cout << *(--ret) << endl;
    string sum = accumulate(vecChar.begin(), vecChar.end(), string(""));
    cout << "Accumulate the vec: " << sum << endl;
    
    sort(vecChar.begin(), vecChar.end());
    auto end_unique = unique(vecChar.begin(), vecChar.end());
    vecChar.erase(end_unique, vecChar.end());
    
    fill(vecChar.begin(), vecChar.end(), 'A');
    for (auto v : vecChar) cout << v << " ";
    cout << endl;

    sort(vecChar.begin(), vecChar.end());
    for (auto v : vecChar) cout << v << " ";
    cout << endl;
    for_each(vecChar.begin(), vecChar.end(),
            [](const char &s) { cout << s << " "; });
    cout << endl;

}

bool isShorter(const string &s1, const string &s2) {
    return s1.size() < s2.size();
}


void biggies(vector<string> &words, vector<string>::size_type sz) {
    stable_sort(words.begin(), words.end(),
                [](const string &a, const string &b) { return a.size() < b.size(); });
    auto wc = find_if(words.begin(), words.end(), 
            [sz](const string &a) { return a.size() >= sz; });
    for_each(wc, words.end(), [](const string &s) { cout << s << " "; });
}


void testMicrosoft() {
    // structure the init parameter
    vector<vector<int> > vvec(4, vector<int>(2, 0));
    vvec[0][0] = 1;
    vvec[0][1] = 2;
    vvec[1][0] = 1;
    vvec[1][1] = 3;
    vvec[2][0] = 1;
    vvec[2][1] = 4;
    vvec[3][0] = 4;
    vvec[3][1] = 5;

    for (auto vv : vvec) {
        for (auto v : vv) {
            cout << v << " ";
        }
    }
    cout << endl;

    // counter the relation between number and flag.
    map<int, string> strMap;
    for (int i = 0; i != 6; ++i) {
        strMap[i] = "";
    }
    strMap[1] = "0";
    for_each(strMap.begin(), strMap.end(),
            [](const pair<int, string> &pa) { cout << pa.first << " "; });
    cout << endl;
    int flag = 1;
    int flagTemp = 0;
    for (auto vv : vvec) {
        if (vv[0] == flag) {
            strMap[vv[1]] = strMap[vv[0]] + to_string(flagTemp++);
        } else {
            flag = vv[0];
            flagTemp = 0;
            strMap[vv[1]] = strMap[vv[0]] + to_string(flagTemp++);
        }
    }   
    for_each(strMap.begin(), strMap.end(),
            [](const pair<int, string> &pa) { cout << pa.second; });
    cout << endl;
}


string hexDi(char s) {
    if (s - '0' < 10) {
        return to_string(s - '0');
    } else {
        return to_string(s - 'A' + 10);
    }
}


string hexAdd(string str1, string str2) {
    int i = 0;
    int c = 0;
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
    string res = "";
    while (i < str1.size() || i < str2.size() || c) {
        int str1Int = i < str1.size() ? str1[i] - '0': 0;
        int str2Int = i < str2.size() ? str2[i] - '0': 0;
        int temp = str1Int + str2Int + c;
        res += to_string(temp % 10);
        c = temp / 10;
        i++;
    }
    reverse(res.begin(), res.end());
    return res;
}


string hexMul(string str1, string str2) {
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
    string res = "";
    int i = 0;
    for (int i = 0; i < str2.size(); ++i) {
        string temp = "";
        int index = 0;
        int c = 0;
        while (c || index < str1.size()) {
            int intTemp = (str1[index] - '0') * (str2[i] - '0');
            temp += to_string(intTemp % 10);
            c = intTemp / 10;
            index++;
        }
        reverse(temp.begin(), temp.end());
        temp += string(i, '0');
        res = hexAdd(res, temp);
    }
    return res;

}


void testNetEast() {
    string strIn = "AA34BBBACDDDD456272464525622BA";
    string strOut = "";

    for (auto s : strIn) {
        strOut = hexAdd(hexMul(strOut, "16"), hexDi(s));
    }
    cout << strOut << endl;
}


void testMomenta() {
    string strIn = "??***AA*??**";
    string strOut = "";
    bool flag = false;

    for (int i = 0; i < strIn.size(); ++i) {
        if (!flag && strIn[i] == '*') strOut.push_back(strIn[i]);
        if (strIn[i] == '*') {
            flag = true;
        } else {
            strOut.push_back(strIn[i]);
            flag = false;
        }
    }

    cout << strOut << endl;

}

