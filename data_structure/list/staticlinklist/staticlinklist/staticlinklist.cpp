#include "stdio.h"
#include "stdlib.h"
#include "ctype.h"
#include "string.h"
#include "time.h"
#include "math.h"
#include "io.h"

#define OK 1;
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 1000

typedef int Status;
typedef char ElemType;

Status visit(ElemType c) {
	printf("%c ", c);

	return OK;
}

typedef struct {
	ElemType data;
	int cur;
}Component, StaticLinkList[MAXSIZE];

// Init the staticinklist element.
Status InitList(StaticLinkList space) {
	for (int i = 0; i < MAXSIZE - 1; i++) {
		space[i].cur = i + 1;
	}
	space[MAXSIZE - 1].cur = 0;

	return OK;
}

// If the static link list is not empty, retrun the lower of the space[0].
int Malloc_SSL(StaticLinkList space) {
	int i = space[0].cur;
	if (space[0].cur)
		space[0].cur = space[i].cur;

	return i;
}

// Put the kth element of SSL into space[0].
void Free_SSL(StaticLinkList space, int k) {
	space[k].cur = space[0].cur;
	space[0].cur = k;
}

// Length of the Stataic Link List.
Status ListLength(StaticLinkList L) {
	int j = 0;
	int i = L[MAXSIZE - 1].cur;
	if (i) {
		j++;
		i = L[i].cur;
	}
	return j;
}

// Insert the element to StaticLinkList.
Status ListInsert(StaticLinkList L, int i, ElemType e) {
	int k = MAXSIZE - 1;
	if (i < 1 || i > ListLength(L) + 1)
		return ERROR;
	int j = Malloc_SSL(L);
	if (j) {
		L[j].data = e;
		for (int l = 1; l < i; l++)
			k = L[k].cur;
		L[j].cur = L[k].cur;
		L[k].cur = j;
		return OK;
	}
	
	return ERROR;
}

// Delete the ith element.
Status ListDelete(StaticLinkList L, int i) {
	int k = MAXSIZE - 1;
	if (i < 1 || i > ListLength(L))
		return ERROR;
	for (int j = 1; j < i; j++) {
		k = L[k].cur;
	}
	int j = L[k].cur;
	L[k].cur = L[j].cur;
	Free_SSL(L, j);

	return OK;
}

//
Status ListTraverse(StaticLinkList L) {
	int k = L[MAXSIZE - 1].cur;
	int j = 0;
	while (k) {
		visit(L[k].data);
		k = L[k].cur;
		j++;
	}
	return j;
	printf("\n");
	return OK;
}

int main() {
	StaticLinkList L;
	Status i;
	i = InitList(L);
	printf("初始化L后：L.length=%d\n", ListLength(L));

	i = ListInsert(L, 1, 'F');
	i = ListInsert(L, 1, 'E');
	i = ListInsert(L, 1, 'D');
	i = ListInsert(L, 1, 'B');
	i = ListInsert(L, 1, 'A');

	printf("\n在L的表头依次插入FEDBA后：\nL.data=");
	ListTraverse(L);

	i = ListInsert(L, 3, 'C');
	printf("\n在L的“B”与“D”之间插入“C”后：\nL.data=");
	ListTraverse(L);

	i = ListDelete(L, 1);
	printf("\n在L的删除“A”后：\nL.data=");
	ListTraverse(L);

	printf("\n");

	return 0;
}