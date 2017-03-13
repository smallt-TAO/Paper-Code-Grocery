#include "stdio.h"
#include "stdlib.h"
#include "io.h"
#include "math.h"
#include "string.h"
#include "ctype.h"
#include "math.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 20

typedef int Status;
typedef int ElemType;

Status visit(ElemType e) {
	printf("%d ", e);
	return OK;
}

typedef struct Node{
	ElemType data;
	struct Node *next;
}Node;
typedef struct Node *LinkList;

// Init the LinkList.
Status InitList(LinkList *L) {
	*L = (LinkList)malloc(sizeof(Node));
	if (!(*L))
		return ERROR;
	(*L)->next = NULL;

	return OK;
}

// Check the linklist if empty.
Status ListEmpty(LinkList *L) {
	if ((*L)->next)
		return FALSE;
	else
		return TRUE;
}

// Clear the LinkList.
Status ClearList(LinkList *L) {
	LinkList p, q;
	p = (*L)->next;
	while (p) {
		q = p->next;
		free(p);
		p = q;
	}
	(*L)->next = NULL;
	return OK;
}

// Return the length of the LinkList.
Status ListLength(LinkList L) {
	int k = 0;
	LinkList p = L->next;
	while (p) {
		k++;
		p = p->next;
	}

	return k;
}

// Use element e to load the ith of the linklist.
Status GetElem(LinkList L, int i, ElemType *e) {
	LinkList p = L->next;
	int j = 1;
	while (p && j < i) {
		j++;
		p = p->next;
	}
	if (!p || j >i)
		return ERROR;
	*e = p->data;

	return OK;
}

int main()
{
	LinkList L;
	ElemType e;
	Status i;
	int j, k;
	i = InitList(&L);
	i = ListEmpty(&L);
	printf("%d", i);
}