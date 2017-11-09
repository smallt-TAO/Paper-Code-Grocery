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
typedef int SElemType;

typedef struct StackNode{
	SElemType data;
	struct StackNode *next;
}StackNode, *LinkStackPtr;

typedef struct {
	LinkStackPtr top;
	int count;
}LinkStack;

Status visit(SElemType e) {
	printf("%d ", e);
	return OK;
}

Status InitStack(LinkStack *S) {
	S->top = (LinkStackPtr)malloc(sizeof(StackNode));
	if (!S->top)
		return ERROR;
	S->top = NULL;
	S->count = 0;
	return OK;
}

Status ClearStack(LinkStack *S) {
	LinkStackPtr p, q;
	p = S->top;
	while (p) {
		q = p;
		p = p->next;
		free(q);
	}
	S->count = 0;
	return OK;
}

Status StackEmpty(LinkStack S) {
	if (S.count == 0)
		return TRUE;
	else
		return FALSE;
}

int StackLength(LinkStack S) {
	return S.count;
}

Status GetTop(LinkStack S, SElemType *e) {
	if (S.top == NULL)
		return ERROR;
	else
		*e = S.top->data;
	return OK;
}

Status Push(LinkStack *S, SElemType e) {
	if (S->count == MAXSIZE)
		return ERROR;
	
	LinkStackPtr p = (LinkStackPtr)malloc(sizeof(StackNode));
	p->data = e;
	p->next = S->top;
	S->top = p;
	S->count++;
	return OK;
}

Status Pop(LinkStack *S, SElemType *e) {
	if (S->count == 0)
		return ERROR;
	
	*e = S->top->data;
	LinkStackPtr p = S->top;
	S->top = S->top->next;
    free(p);
	S->count--;
	
	return OK;
}

Status StackTraverse(LinkStack S) {
	LinkStackPtr p = S.top;
	while (p) {
		visit(p->data);
		p = p->next;
	}
	printf("\n");
	return OK;
}

int main()
{
	int j;
	LinkStack s;
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