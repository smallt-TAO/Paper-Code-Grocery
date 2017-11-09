#include "stdio.h"
#include "stdlib.h"
#include "io.h"
#include "time.h"
#include "math.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0 
#define MAXSIZE 20

typedef int Status;
typedef int QElemType;
typedef struct QNode {
	QElemType data;
	struct QNode *next;
}QNode, *QueuePtr;

typedef struct {
	QueuePtr front;
	QueuePtr rear;
}LinkQueue;

Status visit(QElemType c) {
	printf("%d ", c);
	return OK;
}

Status InitQueue(LinkQueue *Q) {
	Q->front = Q->rear = (QueuePtr)malloc(sizeof(QNode));
	if (!Q->front)
		exit(OVERFLOW);
	Q->front->next = NULL;

	return OK;
}

Status DestroyQueue(LinkQueue *Q) {
	while (Q->front) {
		Q->rear = Q->front->next;
		free(Q->front);
		Q->front = Q->rear;
	}

	return OK;
}

Status ClearQueue(LinkQueue *Q) {
	QueuePtr p, q;
	Q->rear = Q->front;
	p = Q->front->next;
	Q->front->next = NULL;
	while (p) {
		q = p;
		p = p->next;
		free(q);
	}

	return OK;
}

Status QueueEmpty(LinkQueue Q) {
	if (Q.rear == Q.front)
		return TRUE;
	else
		return FALSE;
}

int QueueLength(LinkQueue Q) {
	int i = 0;
	QueuePtr p = Q.front;
	while (Q.rear != p) {
		p = p->next;
		i++;
	}
	return i;
}

Status GetHead(LinkQueue Q, QElemType *e) {
	if (Q.front == Q.rear)
		return ERROR;
	*e = Q.front->next->data;
	return OK;
}

Status EnQueue(LinkQueue *Q, QElemType e) {
	QueuePtr p = (QueuePtr)malloc(sizeof(QNode));
	if (!p)
		return ERROR;
	p->data = e;
	p->next = NULL;
	Q->rear->next = p;
	Q->rear = p;
	return OK;
}

Status DeQueue(LinkQueue *Q, QElemType *e) {
	if (Q->front == Q->rear)
		return ERROR;
	QueuePtr p = Q->front->next;
	*e = p->data;
	Q->front->next = p->next;
	if (Q->rear == p)
		Q->rear = Q->front;
	free(p);
	return OK;
}

Status QueueTraverse(LinkQueue Q) {
	QueuePtr p = Q.front->next;
	while (p) {
		visit(p->data);
		p = p->next;
	}
	printf("\n");
	return OK;
}

int main() {
	int i;
	QElemType d;
	LinkQueue q;
	i = InitQueue(&q);
	if (i)
		printf("�ɹ��ع�����һ���ն���!\n");
	printf("�Ƿ�ն��У�%d(1:�� 0:��)  ", QueueEmpty(q));
	printf("���еĳ���Ϊ%d\n", QueueLength(q));
	EnQueue(&q, -5);
	EnQueue(&q, 5);
	EnQueue(&q, 10);
	printf("����3��Ԫ��(-5,5,10)��,���еĳ���Ϊ%d\n", QueueLength(q));
	printf("�Ƿ�ն��У�%d(1:�� 0:��)  ", QueueEmpty(q));
	printf("���е�Ԫ������Ϊ��");
	QueueTraverse(q);
	i = GetHead(q, &d);
	if (i == OK)
		printf("��ͷԪ���ǣ�%d\n", d);
	DeQueue(&q, &d);
	printf("ɾ���˶�ͷԪ��%d\n", d);
	i = GetHead(q, &d);
	if (i == OK)
		printf("�µĶ�ͷԪ���ǣ�%d\n", d);
	ClearQueue(&q);
	printf("��ն��к�,q.front=%u q.rear=%u q.front->next=%u\n", q.front, q.rear, q.front->next);
	DestroyQueue(&q);
	printf("���ٶ��к�,q.front=%u q.rear=%u\n", q.front, q.rear);

	return 0;
}