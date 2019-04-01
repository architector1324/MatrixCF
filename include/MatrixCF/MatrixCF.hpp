#pragma once


#include "EasyCL.hpp"

namespace mcf{
    using namespace ecl;
    static Program methods = Program::loadProgram("matrix.cl");

    template<typename T>
    class Mat{
    private:
        Variable<size_t> h, w, total_size;
        Array<T> array;
        Variable<bool> ref;
    public:
        Mat();

        const T* getConstArray() const;
        T* getArray();
        size_t getH() const;
        size_t getW() const;
        size_t getTotalSize() const;

        // ~Mat();
    };
}

// IMPLEMENTATION

template<typename T>
mcf::Mat<T>::Mat(){
    h = 0;
    w = 0;
    total_size = 0;
    array = Array<T>(nullptr, 0);
    ref = false;
}

template<typename T>
const T* mcf::Mat<T>::getConstArray() const{
    return array.getConstArray();
}
template<typename T>
T* mcf::Mat<T>::getArray(){
    return array.getArray();
}

template<typename T>
size_t mcf::Mat<T>::getH() const{
    return h.getValue();
}
template<typename T>
size_t mcf::Mat<T>::getW() const{
    return w.getValue();
}
template<typename T>
size_t mcf::Mat<T>::getTotalSize() const{
    return total_size.getValue();
}