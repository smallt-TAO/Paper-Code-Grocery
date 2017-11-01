#ifndef TEMPLATE_DEMO_HXX
#define TEMPLATE_DEMO_HXX

template<class T, int MAXSIZE> class Stack {
private:
	T elems[MAXSIZE];
	int numElems;
public:
	Stack();  // Constrate the function.
	void push(T const&);
	void pop();
	T top() const;

	bool empty() const {
		return numElems == 0;
	}

	bool full() const {
		return numElems == MAXSIZE;
	}
};


template<class T, int MAXSIZE> Stack<T, MAXSIZE>::Stack() :numElems(0) {
	std::cout << "This is the stack. " << std::endl;
}

template<class T, int MAXSIZE>
void Stack<T, MAXSIZE>::push(T const& elem) {
	if (numElems == MAXSIZE)
		throw std::out_of_range("Stack<> ::push() : stack is full");

	elems[numElems++] = elem;
}

template<class T, int MAXSIZE>
void Stack<T, MAXSIZE>::pop() {
	if (numElems <= 0)
		throw std::out_of_range("Stack<> ::pop() : stack is empty");

	numElems--;
}

template<class T, int MAXSIZE>
T Stack<T, MAXSIZE>::top() const {
	if (numElems <= 0)
		throw std::out_of_range("Stack<> ::top() : stack is empty");

	return elems[numElems - 1];
}

#endif