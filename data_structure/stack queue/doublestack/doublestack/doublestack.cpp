#include "stdio.h"
#include "time.h"
#include "math.h"
#include "io.h"
#include "stdlib.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 20

typedef int Status;
typedef int SElemType;

typedef struct {
	SElemType data[MAXSIZE];
	int top1;
	int top2;
}SqDoubleStack;

Status visit(SElemType e) {
	printf("%d ", e);
	return OK;
}

Status InitStack(SqDoubleStack *S) {
	S->top1 = -1;
	S->top2 = MAXSIZE;
	return OK;
}

Status StackLength(SqDoubleStack S) {
	return S.top1 + 1 + MAXSIZE - S.top2;
}

Status ClearStack(SqDoubleStack *S) {
	S->top1 = -1;
	S->top2 = MAXSIZE;
	return OK;
}

Status StackEmpty(SqDoubleStack S) {
	if (S.top1 == -1 && S.top2 == MAXSIZE)
		return TRUE;
	else
		return FALSE;
}

Status StackTraverse(SqDoubleStack S) {
	if (S.top1 == -1 && S.top2 == MAXSIZE)
		return ERROR;
	int i = 0;
	while (i <= S.top1) {
		visit(S.data[i++]);
	}
	i = S.top2;
	while (i < MAXSIZE) {
		visit(S.data[i++]);
	}

	printf("\n");
	return OK;
}

Status Push(SqDoubleStack *S, SElemType e, int stackNumber) {
	if (S->top1 + 1 == S->top2)
		return ERROR;
	if (stackNumber == 1)
		S->data[++S->top1] = e;
	else if (stackNumber == 2)
		S->data[--S->top2] = e;

	return OK;
}

Status Pop(SqDoubleStack *S, SElemType *e, int stackNumber) {
	if (S->top1 == -1 && S->top2 == MAXSIZE)
		return ERROR;
	if (stackNumber == 1)
		*e = S->data[S->top1--];
	else if (stackNumber == 2)
		*e = S->data[S->top2++];

	return OK;
}

int main()
{
	int j;
	SqDoubleStack s;
	int e;
	if (InitStack(&s) == OK) {
		for (j = 1; j <= 5; j++)
			Push(&s, j, 1);
		for (j = MAXSIZE; j >= MAXSIZE - 2; j--)
			Push(&s, j, 2);
	}

	printf("ջ��Ԫ������Ϊ��");
	StackTraverse(s);

	printf("��ǰջ��Ԫ���У�%d \n", StackLength(s));

	Pop(&s, &e, 2);
	printf("������ջ��Ԫ�� e=%d\n", e);
	printf("ջ�շ�%d(1:�� 0:��)\n", StackEmpty(s));

	for (j = 6; j <= MAXSIZE - 2; j++)
		Push(&s, j, 1);

	printf("ջ��Ԫ������Ϊ��");
	StackTraverse(s);

	printf("ջ����%d(1:�� 0:��)\n", Push(&s, 100, 1));


	ClearStack(&s);
	printf("���ջ��ջ�շ�%d(1:�� 0:��)\n", StackEmpty(s));

	return 0;
}
