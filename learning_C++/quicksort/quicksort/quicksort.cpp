#include <iostream>

using namespace std;

void quicksort(int a[], int, int);

int main() {
	int a[] = { 1, 3, 9, 4, 0 };
	int len = sizeof(a) / sizeof(int);
	
	for (int i = 0; i < len; i++)
		cout << a[i] << ",";
	cout << endl;

	quicksort(a, 0, len - 1);

	for (int i = 0; i < len; i++)
		cout << a[i] << ",";
	cout << endl;

	system("pause");
	return 0;
}

void quicksort(int a[], int l, int r) {
	if (l < r) {
		int i = l;
		int j = r;
		int tmp = a[l];
		while (i < j) {
			while (i < j && a[j] <= tmp)
				j--;
			if (i < j)
				a[i++] = a[j];
			while (i < j && a[i] > tmp)
				i++;
			if (i < j)
				a[j--] = a[i];
		}
		a[i] = tmp;
		quicksort(a, l, i - 1);
		quicksort(a, i + 1, r);
	}
}