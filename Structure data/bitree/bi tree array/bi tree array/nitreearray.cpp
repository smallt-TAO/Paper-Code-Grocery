#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "io.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 100
#define MAX_TREE_SIZE 100

typedef int Status;
typedef int TElemType;
typedef TElemType SqBiTree[MAX_TREE_SIZE];

typedef struct {
	int level, order;
}Position;

TElemType Nil = 0;

Status visit(TElemType c) {
	printf("%d ", c);
	return OK;
}

// Init the empty Binary tree.
Status InitBiTree(SqBiTree T) {
	for (int i = 0; i < MAX_TREE_SIZE; i++) {
		T[i] = Nil;
	}
	return OK;
}

Status CreateBiTree(SqBiTree T) {
	int i = 0;
	while (i < 10) {
		T[i] = i + 1;
		if (i != 0 && T[(i + 1) / 2 - 1] == Nil && T[i] != Nil) {
			printf("Have the error point %d.", T[i]);
			exit(ERROR);
		}
		i++;
	}
	while (i < MAX_TREE_SIZE) {
		T[i] = Nil;
		i++;
	}
	return OK;
}

Status BiTreeEmpty(SqBiTree T) {
	if (T[0] == Nil)
		return TRUE;
	else
		return FALSE;
}

Status BiTreeDepth(SqBiTree T) {
	int i;
	int j = -1;
	for (i = 0; i < MAX_TREE_SIZE; i++) {
		if (T[i] != Nil)
			break;
	}
	i++;
	do {
		j++;
	} while (i >= pow((double)2, (double)j));
	return j;
}

Status Root(SqBiTree T, TElemType *e) {
	if (BiTreeEmpty(T))
		return ERROR;
	else {
		*e = T[0];
		return OK;
	}
}

Status Value(SqBiTree T, Position e) {
	return T[(int)powl(2, e.level - 1) + e.order - 2];
}

Status Assign(SqBiTree T, Position e, TElemType value) {
	int i = (int)powl(2, e.level - 1) + e.order - 2;
	if (value != Nil && T[(i + 1) / 2 - 1] == Nil)
		return ERROR;
	else if (value == Nil && T[i * 2 + 1] != Nil || T[i * 2 + 2] != Nil)
		return ERROR;
	T[i] = value;
	return OK;
}

int main()
{
	Status i;
	Position p;
	TElemType e;
	SqBiTree T;
	InitBiTree(T);
	CreateBiTree(T);
	printf("建立二叉树后,树空否？%d(1:是 0:否) 树的深度=%d\n", BiTreeEmpty(T), BiTreeDepth(T));
	i = Root(T, &e);
	if (i)
		printf("二叉树的根为：%d\n", e);
	else
		printf("树空，无根\n");
}