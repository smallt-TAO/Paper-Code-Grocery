#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "io.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 20

typedef int Status;
typedef int ElemType;

Status visit(ElemType c) {
	printf("%d ", c);
	return OK;
}

typedef struct{
	ElemType data[MAXSIZE];
	int length;
}SqList;

// Init the SqList
Status InitList(SqList *L) {
	L->length = 0;
	return OK;
}

// Check the Sqlist if empty.
Status ListEmpty(SqList L) {
	if (L.length == 0)
		return TRUE;
	else
		return FALSE;
}

// Reset the SqList empty.
Status ListClear(SqList *L) {
	L->length = 0;
	return OK;
}

// Return the length of the SqList.
Status ListLength(SqList L) {
	return L.length;
}

// Use element e to return the ith element in SqList.
Status GetElem(SqList L, int i, ElemType *e) {
	if (L.length == 0 || i < 1 || i > L.length)
		return ERROR;
	*e = L.data[i - 1];

	return OK;
}

// If e in SqList, return locate of the same number. Otherwise return 0.
Status LocateElem(SqList L, ElemType e) {
	if (L.length == 0)
		return ERROR;
	for (int i = 0; i < L.length; i++) {
		if (L.data[i] == e)
			return i + 1;
	}

	return 0;
}

// Insert the element e in ith of the SqList.
Status ListInsert(SqList *L, int i, ElemType e) {
	if (L->length == MAXSIZE)
		return ERROR;
	if (i < 1 || i > L->length + 1)
		return ERROR;
	if (i <= L->length) {
		for (int k = L->length - 1; k >= i - 1; k--)
			L->data[k + 1] = L->data[k];
	}
	L->data[i - 1] = e;
	L->length++;

	return OK;
}

// Delete the ith element of the SqList.
Status ListDelete(SqList *L, int i, ElemType *e) {
	if (L->length == 0 || L->length == MAXSIZE)
		return ERROR;
	if (i < 1 || i > L->length)
		return ERROR;
	*e = L->data[i - 1];
	for (int k = i - 1; k <= L->length - 1; k++)
		L->data[k] = L->data[k + 1];
	L->length--;

	return OK;
}

// Traverse the all element of the SqList.
Status ListTraverse(SqList L) {
	if (L.length == 0)
		return ERROR;
	for (int i = 0; i < L.length; i++)
		visit(L.data[i]);
	printf("\n");

	return OK;
}

// Union two SqList.
Status UnionL(SqList *La, SqList Lb) {
	ElemType e;
	int len_a = ListLength(*La);
	int len_b = ListLength(Lb);
	for (int i = 1; i <= len_b; i++) {
		GetElem(Lb, i, &e);
		if (!LocateElem(*La, e))
			ListInsert(La, ++len_a, e);
	}

	return OK;
}

int main() {
	SqList L;
	SqList Lb;

	ElemType e;
	Status i;
	int j, k;
	i = InitList(&L);
	printf("初始化L后：L.length=%d\n", L.length);
	for (j = 1; j <= 5; j++)
		i = ListInsert(&L, 1, j);
	printf("在L的表头依次插入1～5后：L.data=");
	ListTraverse(L);

	printf("L.length=%d \n", L.length);
	i = ListEmpty(L);
	printf("L是否空：i=%d(1:是 0:否)\n", i);

	i = ListClear(&L);
	printf("清空L后：L.length=%d\n", L.length);
	i = ListEmpty(L);
	printf("L是否空：i=%d(1:是 0:否)\n", i);

	for (j = 1; j <= 10; j++)
		ListInsert(&L, j, j);
	printf("在L的表尾依次插入1～10后：L.data=");
	ListTraverse(L);

	printf("L.length=%d \n", L.length);

	ListInsert(&L, 1, 0);
	printf("在L的表头插入0后：L.data=");
	ListTraverse(L);
	printf("L.length=%d \n", L.length);

	GetElem(L, 5, &e);
	printf("第5个元素的值为：%d\n", e);
	for (j = 3; j <= 4; j++) {
		k = LocateElem(L, j);
		if (k)
			printf("第%d个元素的值为%d\n", k, j);
		else
			printf("没有值为%d的元素\n", j);
	}


	k = ListLength(L); /* k为表长 */
	for (j = k + 1; j >= k; j--) {
		i = ListDelete(&L, j, &e); /* 删除第j个数据 */
		if (i == ERROR)
			printf("删除第%d个数据失败\n", j);
		else
			printf("删除第%d个的元素值为：%d\n", j, e);
	}
	printf("依次输出L的元素：");
	ListTraverse(L);

	j = 5;
	ListDelete(&L, j, &e); /* 删除第5个数据 */
	printf("删除第%d个的元素值为：%d\n", j, e);

	printf("依次输出L的元素：");
	ListTraverse(L);

	//构造一个有10个数的Lb
	i = InitList(&Lb);
	for (j = 6; j <= 15; j++)
		i = ListInsert(&Lb, 1, j);

	UnionL(&L, Lb);

	printf("依次输出合并了Lb的L的元素：");
	ListTraverse(L);

	return 0;
}