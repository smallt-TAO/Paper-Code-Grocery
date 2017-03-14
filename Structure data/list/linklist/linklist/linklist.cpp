#include "stdio.h"
#include "stdlib.h"
#include "io.h"
#include "math.h"
#include "string.h"
#include "ctype.h"
#include "time.h"

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

// Find whether there be the same element in LinkList.
Status LocateElem(LinkList L, int i) {
	if (!L->next)
		return ERROR;
	LinkList p = L->next;
	int j = 1;
	while (p) {
		if (p->data == i)
			return j;
		j++;
		p = p->next;
	}

	return FALSE;
}

// Insert the data for LinkList.
Status ListInsert(LinkList *L, int i, ElemType e) {
	LinkList p = *L;
	LinkList s;
	int j = 1;
	while (p && j < i) {
		p = p->next;
		j++;
	}
	if (!p || j > i)
		return ERROR;
	s = (LinkList)malloc(sizeof(Node));
	s->data = e;
	s->next = p->next;
	p->next = s;

	return OK;
}

// Delete the ith element, and return the number.
Status ListDelete(LinkList *L, int i, ElemType *e) {
	LinkList p = *L;
	LinkList q;
	int j = 1;
	while (p->next && j < i) {
		p = p->next;
		j++;
	}
	if (j > i || !(p->next))
		return ERROR;
	q = p->next;
	*e = q->data;
	p->next = q->next;
	free(q);

	return OK;
}

// Create the linklist through the head
void CreateListHead(LinkList *L, int n) {
	LinkList p;
	srand(time(0));
	(*L) = (LinkList)malloc(sizeof(Node));
	(*L)->next = NULL;
	for (int i = 0; i < n; i++) {
		p = (LinkList)malloc(sizeof(Node));
		p->data = rand() % 100 + 1;
		p->next = (*L)->next;
		(*L)->next = p;
	}
}

// Create the linklist through the tail.
void CreateListTail(LinkList *L, int n) {
	LinkList p, r;
	srand(time(0));
	*L = (LinkList)malloc(sizeof(Node));
	r = *L;
	for (int i = 0; i < n; i++) {
		p = (LinkList)malloc(sizeof(Node));
		p->data = rand() % 100 + 1;
		r->next = p;
		r = r->next;
	}
	r->next = NULL;
}

// Traverse the all element of LinkList.
Status ListTraverse(LinkList *L) {
	LinkList p = (*L)->next;
	while (p) {
		visit(p->data);
		p = p->next;
	}
	printf("\n");

	return OK;
}

int main()
{
	LinkList L;
	ElemType e;
	Status i;
	int j, k;
	i = InitList(&L);
	printf("��ʼ��L��ListLength(L)=%d\n", ListLength(L));
	for (j = 1; j <= 5; j++)
		i = ListInsert(&L, 1, j);
	printf("��L�ı�ͷ���β���1��5��L.data=");
	ListTraverse(&L);

	printf("ListLength(L)=%d \n", ListLength(L));
	i = ListEmpty(&L);
	printf("L�Ƿ�գ�i=%d(1:�� 0:��)\n", i);

	i = ClearList(&L);
	printf("���L��ListLength(L)=%d\n", ListLength(L));
	i = ListEmpty(&L);
	printf("L�Ƿ�գ�i=%d(1:�� 0:��)\n", i);

	for (j = 1; j <= 10; j++)
		ListInsert(&L, j, j);
	printf("��L�ı�β���β���1��10��L.data=");
	ListTraverse(&L);

	printf("ListLength(L)=%d \n", ListLength(L));

	ListInsert(&L, 1, 0);
	printf("��L�ı�ͷ����0��L.data=");
	ListTraverse(&L);
	printf("ListLength(L)=%d \n", ListLength(L));

	GetElem(L, 5, &e);
	printf("��5��Ԫ�ص�ֵΪ��%d\n", e);
	for (j = 3; j <= 4; j++) {
		k = LocateElem(L, j);
		if (k)
			printf("��%d��Ԫ�ص�ֵΪ%d\n", k, j);
		else
			printf("û��ֵΪ%d��Ԫ��\n", j);
	}


	k = ListLength(L); /* kΪ�� */
	for (j = k + 1; j >= k; j--) {
		i = ListDelete(&L, j, &e); /* ɾ����j������ */
		if (i == ERROR)
			printf("ɾ����%d������ʧ��\n", j);
		else
			printf("ɾ����%d����Ԫ��ֵΪ��%d\n", j, e);
	}
	printf("�������L��Ԫ�أ�");
	ListTraverse(&L);

	j = 5;
	ListDelete(&L, j, &e); /* ɾ����5������ */
	printf("ɾ����%d����Ԫ��ֵΪ��%d\n", j, e);

	printf("�������L��Ԫ�أ�");
	ListTraverse(&L);

	i = ClearList(&L);
	printf("\n���L��ListLength(L)=%d\n", ListLength(L));
	CreateListHead(&L, 20);
	printf("���崴��L��Ԫ��(ͷ�巨)��");
	ListTraverse(&L);

	i = ClearList(&L);
	printf("\nɾ��L��ListLength(L)=%d\n", ListLength(L));
	CreateListTail(&L, 20);
	printf("���崴��L��Ԫ��(β�巨)��");
	ListTraverse(&L);

	return 0;
}