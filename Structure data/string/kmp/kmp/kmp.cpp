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