#include "stdlib.h"
#include "stdio.h"
#include "io.h"
#include "math.h"
#include "time.h"
#include "string.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0

#define MAXSIZE 100

typedef int Status;
typedef int SElemType;
typedef char String[MAXSIZE + 1];

Status StrAssign(String T, char *chars) {
	if (strlen(chars) > MAXSIZE)
		return ERROR;
	for (int i = 1; i <= MAXSIZE; i++) {
		T[i] = *(chars + i - 1);
	}
	T[0] = strlen(chars);
	return OK;
}

Status ClearString(String S) {
	S[0] = 0;
	return OK;
}

void StrPrint(String S) {
	for (int i = 1; i <= S[0]; i++)
		printf("%d ", S[i]);
	printf("\n");
}

void NextPrint(int next[], int length) {
	for (int i = 1; i <= length; i++)
		printf("%d ", next[i]);
	printf("\n");
}

int StrLength(String S) {
	return S[0];
}

