#include <iostream>
using namespace std;

class A {
public:
	A() : a(10) {};
	virtual void g() { cout << "A()::g() " << a << endl; }
	int a;
};

class B :public A {
public:
	B() :b(20){};
	virtual void g() { cout << "B()::g() " << b << endl; }
	int b;
};

void main() {
	A a;
	B b;
	B *p = &b;
	p->g();
	// a = b;
	// a.g();
}