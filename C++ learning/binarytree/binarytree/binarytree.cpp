#include <iostream>

using namespace std;

struct BiNode{
	char data;
	BiNode *lchild, *rchild;
}*BiTree;

int NodeID;

BiNode *CreateBiTree(char *c, int n);
void PreOrderTraverse(BiNode *T);

int main() {
	int num;
	char a[100];
	cin >> num;
	for (int i = 1; i <= num; i++)
		cin >> a[i];

	NodeID = 0;
	BiTree = CreateBiTree(a, num);
	PreOrderTraverse(BiTree);
	cout << endl;

	return 0;
}

BiNode *CreateBiTree(char *c, int n) {
	BiNode* T;
	NodeID++;
	if (NodeID > n)
		return (NULL);
	if (c[NodeID] == '0')
		return (NULL);
	T = new BiNode;
	T->data = c[NodeID];
	T->lchild = CreateBiTree(c, n);
	T->rchild = CreateBiTree(c, n);
	return (T);
}

void PreOrderTraverse(BiNode *T) {
	if (T) {
		PreOrderTraverse(T->lchild);
		cout << T->data << " ";
		
		PreOrderTraverse(T->rchild);
	}
}