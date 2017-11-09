#include "stdio.h"
#include "stdlib.h"
#include "io.h"
#include "math.h"
#include "time.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 20

typedef int Status;
typedef int SElemType;
typedef struct {
	SElemType data[MAXSIZE];
	int top;
}SqStack;

// Visit the Statck.
Status visit(SElemType c) {
	printf("%d ", c);

	return OK;
}

// Init Stack
Status InitStack(SqStack *s) {
	(*s).top = -1;
	return OK;
}

// Clear the stack
Status ClearStack(SqStack *s) {
	(*s).top = -1;
	return OK;
}

// Check the stack whether empty.
Status StackEmpty(SqStack s) {
	if (s.top == -1)
		return TRUE;
	else
		return FALSE;
}

// return the length of the stack
Status StackLength(SqStack s) {
	return s.top + 1;
}

// use e to return the element of the top of stack
Status GetTop(SqStack S, SElemType *e) {
	if (S.top == -1)
		return ERROR;
	return *e = S.data[S.top];
	return OK;
}

// Push the element to the top.
Status Push(SqStack *S, SElemType e) {
	if (S->top == MAXSIZE - 1)
		return ERROR;
	S->top++;
	S->data[S->top] = e;
	return OK;
}

// Pop the top element of the stack
Status Pop(SqStack *S, SElemType *e) {
	if (S->top == -1)
		return ERROR;
	*e = S->data[S->top];
	S->top--;
	return OK;
}

Status StackTraverse(SqStack S) {
	if (S.top == -1)
		return ERROR;
	int i = 0;
	while (i <= S.top) {
		visit(S.data[i++]);
	}
	printf("\n");
	return OK;
}

int main()
{
	int j;
	SqStack s;
	int e;
	if (InitStack(&s) == OK)
	for (j = 1; j <= 10; j++)
		Push(&s, j);
	printf("Õ»ÖÐÔªËØÒÀ´ÎÎª£º");
	StackTraverse(s);
	Pop(&s, &e);
	printf("µ¯³öµÄÕ»¶¥ÔªËØ e=%d\n", e);
	printf("Õ»¿Õ·ñ£º%d(1:¿Õ 0:·ñ)\n", StackEmpty(s));
	GetTop(s, &e);
	printf("Õ»¶¥ÔªËØ e=%d Õ»µÄ³¤¶ÈÎª%d\n", e, StackLength(s));
	ClearStack(&s);
	printf("Çå¿ÕÕ»ºó£¬Õ»¿Õ·ñ£º%d(1:¿Õ 0:·ñ)\n", StackEmpty(s));

	return 0;
}
