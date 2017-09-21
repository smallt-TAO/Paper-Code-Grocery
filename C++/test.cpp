#include "math.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <stack>

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
    testHeap();
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












