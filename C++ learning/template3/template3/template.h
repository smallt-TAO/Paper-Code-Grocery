#ifndef TEMPLATE_DEMO_HXX
#define TEMPLATE_DEMO_HXX

template<class T1, class T2, class T3 = int> class SumDemo{
public:
	double Cell(T1 a, T2 b, T3 c);
};

template<class T1, class T2, class T3>
double SumDemo<T1, T2, T3>::Cell(T1 a, T2 b, T3 c) {
	return (a + b + c);
}

#endif