#include <iostream>

using namespace std;

template<class Type> class BSTree;

template<class Type> class BinaryNode {
	friend class BSTree<Type>;
private:
	Type data;
	BinaryNode *left;
	BinaryNode *right;
public:
	BinaryNode() : left(NULL), right(NULL) {}
	BinaryNode(const Type& value) : data(value), left(NULL), right(NULL) {}
};

template<class Type> class BSTree {
	BinaryNode<Type> *root;
public:
	BSTree() : root(NULL) {}
	BinaryNode<Type> * GetRoot() const { return root; }
	Type AddValue(const Type& value);
	void printData_NLR(const BinaryNode<Type> * startNode);
	void printData_LNR(const BinaryNode<Type> * startNode);
	void printData_LRN(const BinaryNode<Type> * startNode);
	void Cout() {}
};

template<class Type> 
Type BSTree<Type>::AddValue(const Type& value) {
	if (root == NULL) {
		root = new BinaryNode<Type>(value);
	}
	else {
		BinaryNode<Type> *node = root;
		while (1) {
			if (value > node->data) {
				if (node->right == NULL) {
					node->right = new BinaryNode<Type>(value);
					break;
				}
				else {
					node = node->right;
				}
			}
			else {
				if (node->left == NULL) {
					node->left = new BinaryNode<Type>(value);
					break;
				}
				else {
					node = node->left;
				}
			}
		}
	}
	return value;
}

template<class Type>
void BSTree<Type>::printData_NLR(const BinaryNode<Type> * startNode) {
	if (startNode == NULL) {
		return;
	}
	else {
		cout << startNode->data << " ";
		printData_NLR(startNode->left);
		printData_NLR(startNode->right);
	}
}

template<class Type>
void BSTree<Type>::printData_LNR(const BinaryNode<Type> * startNode) {
	if (startNode == NULL) {
		return;
	}
	else {
		printData_LNR(startNode->left);
		cout << startNode->data << " ";
		printData_LNR(startNode->right);
	}
}

template<class Type>
void BSTree<Type>::printData_LRN(const BinaryNode<Type> * startNode) {
	if (startNode == NULL) {
		return;
	}
	else {
		printData_LRN(startNode->left);
		printData_LRN(startNode->right);
		cout << startNode->data << " ";
	}
}

int main(){
	BSTree<int> tree;
	tree.AddValue(7);
	tree.AddValue(4);
	tree.AddValue(10);
	tree.AddValue(1);
	tree.AddValue(5);
	tree.AddValue(-1);
	tree.AddValue(2);
	tree.AddValue(9);
	tree.AddValue(13);
	tree.AddValue(12);
	tree.AddValue(11);
	tree.AddValue(14);

	cout << "前序遍历：根节点 -> 左子树 -> 右子树" << endl;
	tree.printData_NLR(tree.GetRoot());
	cout << "\n\n中序遍历: 左子树-> 根节点  -> 右子树" << endl;
	tree.printData_LNR(tree.GetRoot());
	cout << "\n\n后续遍历: 左子树 -> 右子树 -> 根节点" << endl;
	tree.printData_LRN(tree.GetRoot());

	return 0;
}