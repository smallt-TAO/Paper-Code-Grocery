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
		printf("%c ", S[i]);
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

int Index(String S, String T, int pos) {
	int i = pos;
	int j = 1;
	while (i <= S[0] && j <= T[0]) {
		if (S[i] == T[j]) {
			i++;
			j++;
		}
		else {
			i = i - j + 2;
			j = 1;
		}
	}
	if (j > T[0])
		return i - T[0];
	else
		return 0;
}

void get_next(String T, int *next) {
	int i = 1;
	int j = 0;
	next[1] = 0;
	while (i < T[0]) {
		if (j == 0 || T[i] == T[j]) {
			i++;
			j++;
			next[i] = j;
		}
		else{
			j = next[j];
		}
	}
}

int Index_KMP(String S, String T, int pos) {
	int i = pos;
	int j = 1;
	int next[255];
	get_next(T, next);
	while (i <= S[0] && j <= T[0]) {
		if (S[i] == T[j]) {
			i++;
			j++;
		}
		else {
			j = next[j];
		}
	}
	if (j > T[0])
		return i - T[0];
	else
		return 0;
}

int main()
{
	int i, *p;
	String s1, s2;

	StrAssign(s1, "abcdex");
	printf("子串为: ");
	StrPrint(s1);
	i = StrLength(s1);
	p = (int*)malloc((i + 1)*sizeof(int));
	get_next(s1, p);
	printf("Next为: ");
	NextPrint(p, StrLength(s1));
	printf("\n");

	StrAssign(s1, "abcabx");
	printf("子串为: ");
	StrPrint(s1);
	i = StrLength(s1);
	p = (int*)malloc((i + 1)*sizeof(int));
	get_next(s1, p);
	printf("Next为: ");
	NextPrint(p, StrLength(s1));
	printf("\n");

	StrAssign(s1, "ababaaaba");
	printf("子串为: ");
	StrPrint(s1);
	i = StrLength(s1);
	p = (int*)malloc((i + 1)*sizeof(int));
	get_next(s1, p);
	printf("Next为: ");
	NextPrint(p, StrLength(s1));
	printf("\n");

	StrAssign(s1, "aaaaaaaab");
	printf("子串为: ");
	StrPrint(s1);
	i = StrLength(s1);
	p = (int*)malloc((i + 1)*sizeof(int));
	get_next(s1, p);
	printf("Next为: ");
	NextPrint(p, StrLength(s1));
	printf("\n");

	StrAssign(s1, "ababaaaba");
	printf("   子串为: ");
	StrPrint(s1);
	i = StrLength(s1);
	p = (int*)malloc((i + 1)*sizeof(int));
	get_next(s1, p);
	printf("   Next为: ");
	NextPrint(p, StrLength(s1));
	/*
	get_nextval(s1, p);
	printf("NextVal为: ");
	NextPrint(p, StrLength(s1));
	printf("\n");
	*/
	

	StrAssign(s1, "aaaaaaaab");
	printf("   子串为: ");
	StrPrint(s1);
	i = StrLength(s1);
	p = (int*)malloc((i + 1)*sizeof(int));
	get_next(s1, p);
	printf("   Next为: ");
	NextPrint(p, StrLength(s1));
	

	printf("\n");

	StrAssign(s1, "00000000000000000000000000000000000000000000000001");
	printf("主串为: ");
	StrPrint(s1);
	StrAssign(s2, "0000000001");
	printf("子串为: ");
	StrPrint(s2);
	printf("\n");
	printf("主串和子串在第%d个字符处首次匹配（朴素模式匹配算法）\n", Index(s1, s2, 1));
	printf("主串和子串在第%d个字符处首次匹配（KMP算法） \n", Index_KMP(s1, s2, 1));
	// printf("主串和子串在第%d个字符处首次匹配（KMP改良算法） \n", Index_KMP1(s1, s2, 1));

	return 0;
}
