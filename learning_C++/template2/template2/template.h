#ifndef TEMPLATE_DEMO_HXX
#define TEMPLATE_DEMO_HXX

template<class T> class CompareDemo {
public:
	int compare(const T a, const T b);
};

template<class T> 
int CompareDemo<T>::compare(const T a, const T b) {
	if (a > b)
		return 1;
	else if (a < b)
		return -1;
	else
		return 0;
}

#endif