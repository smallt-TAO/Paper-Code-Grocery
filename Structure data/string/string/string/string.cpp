#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "time.h"
#include "io.h"

#define OK 1
#define ERROR 0
#define TRUE 1
#define FALSE 0
#define MAXSIZE 40

typedef int Status;
typedef int ElemType;
typedef char String[MAXSIZE + 1];  // Place 0 save the length of the string.

Status StrAssign(String T, char *chars) {
	if (strlen(chars) > MAXSIZE)
		return ERROR;
	else {
		T[0] = strlen(chars);
		for (int i = 1; i <= T[0]; i++)
			T[i] = *(chars + i - 1);
		return OK;
	}
}

Status StrCopy(String T, String S) {
	for (int i = 0; i <= S[0]; i++)
		T[i] = S[i];
	return OK;
}

Status StrEmpty(String T) {
	if (T[0] == 0)
		return TRUE;
	else
		return FALSE;
}

Status StrCompare(String S, String T) {
	for (int i = 1; i <= T[0] && i <= S[0]; i++) {
		if (T[i] != S[i])
			return S[i] - T[i];
	}

	return S[0] - T[0];
}

int StrLength(String T) {
	return T[0];
}

Status ClearString(String S) {
	S[0] = 0;
	return OK;
}

Status Concat(String T, String S1, String S2) {
	if (S1[0] + S2[0] <= MAXSIZE) {
		for (int i = 1; i <= S1[0]; i++)
			T[i] = S1[i];
		for (int i = 1; i <= S2[0]; i++)
			T[i + S1[0]] = S2[i];
		T[0] = S1[0] + S2[0];
		return TRUE;
	}
	else {
		for (int i = 1; i <= S1[0]; i++)
			T[i] = S1[i];
		for (int i = 1; i <= MAXSIZE - S1[0]; i++)
			T[i + S1[0]] = S2[i];
		T[0] = MAXSIZE;
		return FALSE;
	}
}

Status SubString(String Sub, String S, int pos, int len) {
	if (pos < 1 || pos > S[0] || len < 1 || len > S[0] - pos + 1)
		return ERROR;
	for (int i = 1; i <= len; i++) {
		Sub[i] = S[pos + i - 1];
	}
	Sub[0] = len;
	return OK;
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
	else {
		return 0;
	}
}

int Index2(String S, String T, int pos) {
	int n, m, i;
	String sub;
	if (pos > 0) {
		n = StrLength(S);
		m = StrLength(T);
		i = pos;
		while (i <= n - m + 1) {
			SubString(sub, S, i, m);
			if (StrCompare(sub, T) != 0)
				i++;
			else
				return i;
		}
	}
	return 0;
}

Status StrInsert(String S, int pos, String T) {
	int i;
	if (pos < 1 || pos > S[0] + 1)
		return ERROR;
	if (S[0] + T[0] <= MAXSIZE) {
		for (i = S[0]; i >= pos; i--)
			S[i + T[0]] = S[i];
		for (i = pos; i < pos + T[0]; i++)
			S[i] = T[i - pos + 1];
		S[0] += T[0];
		return TRUE;
	}
	else {
		for (i = MAXSIZE; i <= pos; i--)
			S[i] = S[i - T[0]];
		for (i = pos; i < pos + T[0]; i++)
			S[i] = T[i - pos + 1];
		S[0] = MAXSIZE;
		return FALSE;
	}
}

Status StrDelete(String S, int pos, int len) {
	if (pos < 1 || pos > S[0] - pos + 1 || len < 1)
		return ERROR;
	for (int i = pos + len; i <= S[0]; i++)
		S[i - len] = S[i];
	S[0] -= len;
	return OK;
}

Status Replace(String S, String T, String V) {
	int i = 1;
	if (StrEmpty(T))
		return ERROR;
	do {
		i = Index(S, T, i);
		if (i) {
			StrDelete(S, i, StrLength(T));
			StrInsert(S, i, V);
			i += StrLength(V);
		}
	} while (i);
	return OK;
}

void StrPrint(String T) {
	for (int i = 1; i <= T[0]; i++)
		printf("%c", T[i]);
	printf("\n");
}


int main() {
	int i, j;
	Status k;
	char s;
	String t, s1, s2;
	printf("�����봮s1: ");

	k = StrAssign(s1, "abcd");
	if (!k) {
		printf("��������MAXSIZE(=%d)\n", MAXSIZE);
		exit(0);
	}
	printf("����Ϊ%d ���շ�%d(1:�� 0:��)\n", StrLength(s1), StrEmpty(s1));
	StrCopy(s2, s1);
	printf("����s1���ɵĴ�Ϊ: ");
	StrPrint(s2);
	printf("�����봮s2: ");

	k = StrAssign(s2, "efghijk");
	if (!k) {
		printf("��������MAXSIZE(%d)\n", MAXSIZE);
		exit(0);
	}
	StrPrint(s2);
	printf("\n");
	i = StrCompare(s1, s2);
	if (i<0)
		s = '<';
	else if (i == 0)
		s = '=';
	else
		s = '>';
	printf("��s1%c��s2\n", s);
	k = Concat(t, s1, s2);
	printf("��s1���Ӵ�s2�õ��Ĵ�tΪ: ");
	StrPrint(t);
	if (k == FALSE)
		printf("��t�нض�\n");
	ClearString(s1);
	printf("��Ϊ�մ���,��s1Ϊ: ");
	StrPrint(s1);
	printf("����Ϊ%d ���շ�%d(1:�� 0:��)\n", StrLength(s1), StrEmpty(s1));
	printf("��t���Ӵ�,�������Ӵ�����ʼλ��,�Ӵ�����: ");

	i = 2;
	j = 3;
	printf("%d,%d \n", i, j);

	k = SubString(s2, t, i, j);
	if (k) {
		printf("�Ӵ�s2Ϊ: ");
		StrPrint(s2);
	}
	printf("�Ӵ�t�ĵ�pos���ַ���,ɾ��len���ַ���������pos,len: ");

	i = 4;
	j = 2;
	printf("%d,%d \n", i, j);


	StrDelete(t, i, j);
	printf("ɾ����Ĵ�tΪ: ");
	StrPrint(t);
	i = StrLength(s2) / 2;
	StrInsert(s2, i, t);
	printf("�ڴ�s2�ĵ�%d���ַ�֮ǰ���봮t��,��s2Ϊ:\n", i);
	StrPrint(s2);
	i = Index(s2, t, 1);
	printf("s2�ĵ�%d����ĸ���t��һ��ƥ��\n", i);
	SubString(t, s2, 1, 1);
	printf("��tΪ��");
	StrPrint(t);
	Concat(s1, t, t);
	printf("��s1Ϊ��");
	StrPrint(s1);
	Replace(s2, t, s1);
	printf("�ô�s1ȡ����s2�кʹ�t��ͬ�Ĳ��ص��Ĵ���,��s2Ϊ: ");
	StrPrint(s2);


	return 0;
}