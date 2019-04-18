#pragma once

#ifndef _WIN32
#define MATRIXCF_USE_OPENMP
#endif // _WIN32


#include <functional>
#include <omp.h>
#include <iomanip>

#include "EasyCL.hpp"
#include "json.hpp"

namespace mcf{
    using namespace ecl;

    enum RAVEL {ROW, COLUMN};
    enum REDUCE {FULL, COLUMNS, ROWS};
    enum TRANSPOSE {NONE, FIRST, SECOND, BOTH};

    template<typename T>
    class Mat{
    private:
        size_t h, w, total_size;
        array<T> arr;
        bool ref;

        void clear();
        std::string getTypeName() const;
        void requireMatrixShape(const Mat<T>&, size_t, size_t, const std::string&, bool is_result = false) const;
        void requireMatrixH(size_t, size_t, const std::string&) const;
        void requireTotalSize(const Mat<T>&, size_t, const std::string&) const;

		void copy(const Mat<T>&);
		void move(Mat<T>&);
    public:
        Mat();
        Mat(size_t, size_t);
        Mat(T*, size_t, size_t);

        Mat(const Mat<T>&);
        Mat<T>& operator=(const Mat<T>&);

        Mat(Mat<T>&&);
        Mat<T>& operator=(Mat<T>&&);

        array<T>& getArray();

        const array<T>& getConstArray() const;
        const size_t getH() const;
        const size_t getW() const;
        const size_t getTotalSize() const;

        size_t totalMemoryUsed() const;
        bool isRef() const;

        const T& getE(size_t, size_t) const;
        void setE(const T&, size_t, size_t);

        T* operator[](size_t);
        operator T*();
        operator const T*() const;

        void send(ecl::Computer&, ecl::EXEC sync = SYNC);
        void receive(ecl::Computer&, ecl::EXEC sync = SYNC);
        void release(ecl::Computer&, ecl::EXEC sync = SYNC);
        void grab(ecl::Computer&, ecl::EXEC sync = SYNC);

        void save(const std::string&) const;
        static Mat<T> load(const std::string&);

        template<typename U>
        friend std::ostream& operator<<(std::ostream&, const Mat<U>&);
        template<typename U>
        friend ecl::Computer& operator<<(ecl::Computer&, Mat<U>&);
        template<typename U>
        friend ecl::Computer& operator>>(ecl::Computer&, Mat<U>&);

        // methods (extra)
        bool equals(const Mat<T>&) const;

        void reshape(size_t, size_t);
        void ravel(RAVEL option = ROW);

        // methods (mutable)
        void gen(const std::function<T(size_t, size_t)>&);
        void gen(const std::string&, ecl::Computer&, ecl::EXEC sync = SYNC);

        void full(const T&);
        void full(const T&, ecl::Computer&, ecl::EXEC sync = SYNC);

        void zeros();
        void zeros(ecl::Computer&, ecl::EXEC sync = SYNC);

        void ones();
        void ones(ecl::Computer&, ecl::EXEC sync = SYNC);

        void eye(const T& value = T(1));
        void eye(const T& value, ecl::Computer&, ecl::EXEC sync = SYNC);

        void hstack(const Mat<T>&, const Mat<T>&);
        void hstack(const Mat<T>&, const Mat<T>&, ecl::Computer&, ecl::EXEC sync = SYNC);

        void vstack(const Mat<T>&, const Mat<T>&);
        void vstack(const Mat<T>&, const Mat<T>&, ecl::Computer&, ecl::EXEC sync = SYNC);

        void cpy(const Mat<T>&);
        void view(Mat<T>&);
        
        // higher-order methods (immutable)
        void map(const std::function<T(const T&)>&, Mat<T>&, TRANSPOSE option = NONE) const;
        void map(const std::string&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        void transform(const Mat<T>&, const std::function<T(const T&, const T&)>&, Mat<T>&, TRANSPOSE option = NONE) const;
        void transform(const Mat<T>&, const std::string&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        // methods (immutable)
        void transpose(Mat<T>&) const;
        void transpose(Mat<T>&, ecl::Computer&, ecl::EXEC sync = SYNC) const;

        void reduce(Mat<T>&, REDUCE option = FULL, TRANSPOSE transpose_option = NONE) const;
        void reduce(Mat<T>&, ecl::Computer&, REDUCE option = FULL, TRANSPOSE transpose_option = NONE, ecl::EXEC sync = SYNC) const;

        T reduce() const;
		T mreduce(const std::function<T(const T&)>&) const;

        void add(const Mat<T>&, Mat<T>&, TRANSPOSE option = NONE) const;
        void add(const Mat<T>&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        void sub(const Mat<T>&, Mat<T>&, TRANSPOSE option = NONE) const;
        void sub(const Mat<T>&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        void hadamard(const Mat<T>&, Mat<T>&, TRANSPOSE option = NONE) const;
        void hadamard(const Mat<T>&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        void mul(const Mat<T>&, Mat<T>&, TRANSPOSE option = NONE) const;
        void mul(const Mat<T>&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        void mul(const T&, Mat<T>&, TRANSPOSE option = NONE) const;
        void mul(const T&, Mat<T>&, ecl::Computer&, TRANSPOSE option = NONE, ecl::EXEC sync = SYNC) const;

        void hsplit(Mat<T>&, Mat<T>&) const;
        void hsplit(Mat<T>&, Mat<T>&, ecl::Computer&, ecl::EXEC sync = SYNC) const;

        void vsplit(Mat<T>&, Mat<T>&) const;
        void vsplit(Mat<T>&, Mat<T>&, ecl::Computer&, ecl::EXEC sync = SYNC) const;

        ~Mat();
    };
}

// IMPLEMENTATION
template<typename T>
void mcf::Mat<T>::clear(){
    h = 0;
    w = 0;
    total_size = 0;
    arr.clear();
    ref = 0;
}

template<typename T>
std::string mcf::Mat<T>::getTypeName() const{
    if constexpr (std::is_same<T, bool>::value) return "bool";
    else if constexpr (std::is_same<T, char>::value) return "char";
    else if constexpr (std::is_same<T, unsigned char>::value) return "unsigned char";
    else if constexpr (std::is_same<T, short>::value) return "short";
    else if constexpr (std::is_same<T, unsigned short>::value) return "unsigned short";
    else if constexpr(std::is_same<T, int>::value) return "int";
    else if constexpr(std::is_same<T, unsigned int>::value) return "unsigned int";
    else if constexpr(std::is_same<T, long>::value) return "long";
    else if constexpr(std::is_same<T, unsigned long>::value) return "unsigned long";
    else if constexpr(std::is_same<T, float>::value) return "float";
    else if constexpr (std::is_same<T, double>::value) return "double";
    else if constexpr(std::is_same<T, size_t>::value) return "size_t";
    else throw std::runtime_error("Get matrix typename: ecl::Computer calculations on matrices with this template aren't supported");
}

template<typename T>
void mcf::Mat<T>::requireMatrixShape(const Mat<T>& X, size_t require_h, size_t require_w, const std::string& where, bool is_result) const{
    size_t r_h = X.getH();
    size_t r_w = X.getW();

    if(r_h != require_h || r_w != require_w){
        std::string what = is_result ? "result matrix" : "matrix";

        std::string e = "Require shape [" + where + "]: ";
        e += "wrong " + what + " shape ";
        e += std::to_string(r_h) + "x" + std::to_string(r_w);
        e += " != ";
        e += std::to_string(require_h) + "x" + std::to_string(require_w);

        throw std::runtime_error(e);
    }
}

template<typename T>
void mcf::Mat<T>::requireMatrixH(size_t r_h, size_t require_h, const std::string& where) const{
    if(r_h != require_h){
        std::string e = "Require h [" + where + "]: ";
        e += "wrong matrix h ";
        e += std::to_string(r_h) + " != " + std::to_string(require_h);
        throw std::runtime_error(e);
    }
}

template<typename T>
void mcf::Mat<T>::requireTotalSize(const Mat<T>& X, size_t require_total_size, const std::string& where) const{
    size_t r_total_size = X.getTotalSize();

    if(r_total_size != require_total_size){
        std::string e = "Require total size [" + where + "]: ";
        e += "wrong matrix total size ";
        e += std::to_string(r_total_size) + " != " + std::to_string(require_total_size);
        throw std::runtime_error(e);
    }
}

template<typename T>
void mcf::Mat<T>::copy(const Mat<T>& other) {
	clear();

	h = other.h;
	w = other.w;
	total_size = other.total_size;
	arr = other.arr;
	ref = false;
}
template<typename T>
void mcf::Mat<T>::move(Mat<T>& other) {
	h = std::move(other.h);
	w = std::move(other.w);
	total_size = std::move(other.total_size);
	arr = std::move(other.arr);
	ref = std::move(other.ref);

	other.clear();
}

// Constructors
template<typename T>
mcf::Mat<T>::Mat(){
    h = 0;
    w = 0;
    total_size = 0;
    ref = false;
}

template<typename T>
mcf::Mat<T>::Mat(size_t h, size_t w) : arr(w * h){
    this->h = h;
    this->w = w;
    total_size = w * h;
    ref = false;
}

template<typename T>
mcf::Mat<T>::Mat(T* arr, size_t h, size_t w) : arr(arr, h * w, READ_WRITE){
    this->h = h;
    this->w = w;
    total_size = w * h;
    ref = true;
}


template<typename T>
mcf::Mat<T>::Mat(const Mat<T>& other){
	copy(other);
}
template<typename T>
mcf::Mat<T>& mcf::Mat<T>::operator=(const Mat<T>& other){
	copy(other);

    return *this;
}

template<typename T>
mcf::Mat<T>::Mat(Mat<T>&& other){
	move(other);
}

template<typename T>
mcf::Mat<T>& mcf::Mat<T>::operator=(Mat<T>&& other){
	move(other);

    return *this;
}

// Getters
template<typename T>
const mcf::array<T>& mcf::Mat<T>::getConstArray() const{
    return arr;
}
template<typename T>
const size_t mcf::Mat<T>::getH() const{
    return h;
}
template<typename T>
const size_t mcf::Mat<T>::getW() const{
    return w;
}
template<typename T>
const size_t mcf::Mat<T>::getTotalSize() const{
    return total_size;
}

template<typename T>
mcf::array<T>& mcf::Mat<T>::getArray(){
    return arr;
}

template<typename T>
size_t mcf::Mat<T>::totalMemoryUsed() const{
    return total_size * sizeof(T);
}
template<typename T>
bool mcf::Mat<T>::isRef() const{
    return ref;
}

template<typename T>
const T& mcf::Mat<T>::getE(size_t i, size_t j) const{
    return arr[w * i + j];
}
template<typename T>
void mcf::Mat<T>::setE(const T& value, size_t i, size_t j){
    arr[w * i + j] = value;
}

template<typename T>
T* mcf::Mat<T>::operator[](size_t i){
    return arr + i * w;
}

template<typename T>
mcf::Mat<T>::operator T*(){
    return arr;
}
template<typename T>
mcf::Mat<T>::operator const T*() const{
    return arr;
}

template<typename T>
void mcf::Mat<T>::send(ecl::Computer& video, ecl::EXEC sync){
    video.send(arr, sync);
}
template<typename T>
void mcf::Mat<T>::receive(ecl::Computer& video, ecl::EXEC sync){
    video.receive(arr, sync);
}
template<typename T>
void mcf::Mat<T>::release(ecl::Computer& video, ecl::EXEC sync){
    video.release(arr, sync);
}
template<typename T>
void mcf::Mat<T>::grab(ecl::Computer& video, ecl::EXEC sync){
    video.grab(arr, sync);
}

template<typename T>
void mcf::Mat<T>::save(const std::string& json_filename) const{
    std::ofstream f(json_filename);
    if(!f.is_open()) throw std::runtime_error("unable to save matrix to json file");

    auto j = nlohmann::json();
    j["w"] = static_cast<size_t>(w);
    j["h"] = static_cast<size_t>(h);
    j["total_size"] = static_cast<size_t>(total_size);
    j["array"] = std::vector<T>(static_cast<const T*>(arr), arr + total_size);

    f << std::setw(4) << j;
    f.close();
}
template<typename T>
mcf::Mat<T> mcf::Mat<T>::load(const std::string& json_filename){
    std::ifstream f(json_filename);
        if(!f.is_open()) throw std::runtime_error("unable to load matrix to json file");

        auto j = nlohmann::json::parse(f);
        T* temp = new T[(size_t)j["total_size"]];
        std::copy(j["array"].begin(), j["array"].end(), temp);

        Mat<T> result(temp, j["h"], j["w"]);
        result.requireTotalSize(result, j["total_size"], "load");

        f.close();

        return std::move(result);
}

namespace mcf{
    template<typename T>
    std::ostream& operator<<(std::ostream& s, const Mat<T>& other){
        for(size_t i = 0; other.h > i; i++){
            s << "(" << other.getE(i, 0);
            for(size_t j = 1; other.w > j; j++) s << ", " << other.getE(i, j);
            s << ")\n";
        }
        return s;
    }

    template<typename T>
    ecl::Computer& operator<<(ecl::Computer& video, Mat<T>& other){
        other.send(video);
        return video;
    }
    template<typename T>
    ecl::Computer& operator>>(ecl::Computer& video, Mat<T>& other){
        other.receive(video);
        return video;
    }
}

// methods (extra)
template<typename T>
bool mcf::Mat<T>::equals(const Mat<T>& X) const{
    try{
        requireMatrixShape(X, h, w, "equals");
    } catch(int){
        return false;
    }

    for(size_t i = 0; total_size > i; i++){
        if(arr[i] != X.arr[i]) return false;
    }
    return true;
}

template<typename T>
void mcf::Mat<T>::reshape(size_t new_h, size_t new_w){
    requireTotalSize(*this, new_h * new_w, "reshape");
    h = new_h;
    w = new_w;
}
template<typename T>
void mcf::Mat<T>::ravel(mcf::RAVEL option){
    if(option == COLUMN) reshape(total_size, 1);
    else reshape(1, total_size);
}

// methods (mutable)
template<typename T>
void mcf::Mat<T>::gen(const std::function<T(size_t, size_t)>& f){
	#ifdef MATRIXCF_USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif 
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++) setE(f(i, j), i, j);
    }
}
template<typename T>
void mcf::Mat<T>::gen(const std::string& body, ecl::Computer& video, ecl::EXEC sync){
    std::string type = getTypeName();

    ecl::Program prog = "__kernel void gen";
    prog += "(__global " + type + "* result)";
    prog += "{\n";
    prog += "size_t i = get_global_id(0);\n";
    prog += "size_t j = get_global_id(1);\n";
    prog += "size_t h = get_global_size(0);\n";
    prog += "size_t w = get_global_size(1);\n";
    prog += "size_t index = i * w + j;\n";
    prog += type + " ret;\n";
    prog += body + "\n";
    prog += "result[index] = ret;";
    prog += "}";

    ecl::Kernel gen = "gen";

    ecl::Frame frame = {prog, gen, {&arr}};
    video.grid(frame, {h, w}, sync);
}

template<typename T>
void mcf::Mat<T>::full(const T& value){
    gen([&](size_t i, size_t j){
        return value;
    });
}
template<typename T>
void mcf::Mat<T>::full(const T& value, ecl::Computer& video, ecl::EXEC sync){
    std::string val = std::to_string(value);
    gen("ret = " + val + ";", video, sync);
}
template<typename T>
void mcf::Mat<T>::zeros(){
    gen([](size_t i, size_t j){
        return T(0);
    });
}
template<typename T>
void mcf::Mat<T>::zeros(ecl::Computer& video, ecl::EXEC sync){
    gen("ret = 0;", video, sync);
}

template<typename T>
void mcf::Mat<T>::ones(){
    gen([](size_t i, size_t j){
        return T(1);
    });
}
template<typename T>
void mcf::Mat<T>::ones(ecl::Computer& video, ecl::EXEC sync){
    gen("ret = 1;", video, sync);
}

template<typename T>
void mcf::Mat<T>::eye(const T& value){
    gen([&](size_t i, size_t j){
        return i == j ? value : T(0);
    });
}
template<typename T>
void mcf::Mat<T>::eye(const T& value, ecl::Computer& video, ecl::EXEC sync){
    std::string val = std::to_string(value);
    gen("ret = i == j ? + " + val + " : 0;", video, sync);
}

template<typename T>
void mcf::Mat<T>::hstack(const Mat<T>& A, const Mat<T>& B){
    requireMatrixH(A.h, B.h, "hstack");
    requireMatrixShape(*this, A.h, A.w + B.w, "hstack", true);

	#ifdef MATRIXCF_USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++){
            if(A.w > j) setE(A.getE(i, j), i, j);
            else setE(B.getE(i, j - A.w), i, j);
        }
    }
}
template<typename T>
void mcf::Mat<T>::hstack(const Mat<T>& A, const Mat<T>& B, ecl::Computer& video, ecl::EXEC sync){
    std::string type = getTypeName();

    requireMatrixH(A.h, B.h, "hstack");
    requireMatrixShape(*this, A.h, A.w + B.w, "hstack", true);

    ecl::Program prog = "__kernel void hstack";
    prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
    prog += "{\n";
    prog += "size_t i = get_global_id(0);\n";
    prog += "size_t j = get_global_id(1);\n";
    prog += "size_t w = get_global_size(1);\n";
    prog += "size_t a_w = " + std::to_string(A.w) + ";\n";
    prog += "size_t b_w = " + std::to_string(B.w) + ";\n";
    prog += "if(j < a_w) result[i * w + j] = a[i * a_w + j];\n";
    prog += "else result[i * w + j] = b[i * b_w + j - a_w];\n";
    prog += "}";

    ecl::Kernel hstack = "hstack";

    ecl::Frame frame = {prog, hstack, {&A.arr, &B.arr, &arr}};
    video.grid(frame, {h, w}, sync);
}

template<typename T>
void mcf::Mat<T>::vstack(const Mat<T>& A, const Mat<T>& B){
    requireMatrixH(A.w, B.w, "vstack");
    requireMatrixShape(*this, A.h + B.h, A.w, "vstack", true);

	#ifdef MATRIXCF_USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++){
            if(A.h > i) setE(A.getE(i, j), i, j);
            else setE(B.getE(i - A.h, j), i, j);
        }
    }
}
template<typename T>
void mcf::Mat<T>::vstack(const Mat<T>& A, const Mat<T>& B, ecl::Computer& video, ecl::EXEC sync){
    std::string type = getTypeName();

    requireMatrixH(A.w, B.w, "vstack");
    requireMatrixShape(*this, A.h + B.h, A.w, "vstack", true);

    ecl::Program prog = "__kernel void vstack";
    prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
    prog += "{\n";
    prog += "size_t i = get_global_id(0);\n";
    prog += "size_t j = get_global_id(1);\n";
    prog += "size_t w = get_global_size(1);\n";
    prog += "size_t a_h = " + std::to_string(A.h) + ";\n";
    prog += "size_t a_w = " + std::to_string(A.w) + ";\n";
    prog += "size_t b_w = " + std::to_string(B.w) + ";\n";
    prog += "if(i < a_h) result[i * w + j] = a[i * a_w + j];\n";
    prog += "else result[i * w + j] = b[(i - a_h) * b_w + j];\n";
    prog += "}";

    ecl::Kernel vstack = "vstack";

    ecl::Frame frame = {prog, vstack, {&A.arr, &B.arr, &arr}};
    video.grid(frame, {h, w}, sync);
}

template<typename T>
void mcf::Mat<T>::cpy(const Mat<T>& X){
    requireMatrixShape(X, h, w, "cpy");

	#ifdef MATRIXCF_USE_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
    for(size_t i = 0; h > i; i++)
        for(size_t j = 0; w > j; j++) setE(X.getE(i, j), i, j);
}
template<typename T>
void mcf::Mat<T>::view(Mat<T>& X){
    requireTotalSize(X, total_size, "view");

	arr.view(X.getArray());
}

// higher-order methods (immutable)
template<typename T>
void mcf::Mat<T>::map(const std::function<T(const T&)>& f, mcf::Mat<T>& result, TRANSPOSE option) const
{
    if(option == NONE){
        requireMatrixShape(result, h, w, "map", true);

		#ifdef MATRIXCF_USE_OPENMP
		#pragma omp parallel for
		#endif
        for(size_t i = 0; total_size > i; i++) result.arr[i] = f(arr[i]);
    }
    else{
        requireMatrixShape(result, w, h, "map", true);

		#ifdef MATRIXCF_USE_OPENMP
		#pragma omp parallel for collapse(2)
		#endif
        for(size_t i = 0; w > i; i++){
            for(size_t j = 0; h > j; j++) result[i][j] = f(getE(j, i));
        }
    }
}
template<typename T>
void mcf::Mat<T>::map(const std::string& body, mcf::Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const
{
    std::string type = getTypeName();

    if(option == NONE){
        requireMatrixShape(result, h, w, "map", true);

        ecl::Program prog = "__kernel void map";
        prog += "(__global " + type + "* a, __global " + type + "* result){\n";
        prog += "size_t index = get_global_id(0);\n";
        prog += type + " v = a[index];\n";
        prog += type + " ret;\n";
        prog += body + "\n";
        prog += "result[index] = ret;\n";
        prog += "}";

        ecl::Kernel map = "map";

        ecl::Frame frame = {prog, map, {&arr, &result.arr}};
        video.grid(frame, {total_size}, sync);
    }
    else{
        requireMatrixShape(result, w, h, "map", true);

        ecl::Program prog = "__kernel void map";
        prog += "(__global " + type + "* a, __global " + type + "* result)";
        prog += "{\n";
        prog += "";
        prog += "size_t result_index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
        prog += "size_t index = get_global_id(1) * get_global_size(0) + get_global_id(0);\n";
        prog += type + " v = a[index];\n";
        prog += type + " ret;\n";
        prog += body + "\n";
        prog += "result[result_index] = ret;\n";
        prog += "}";

        ecl::Kernel map = "map";

        ecl::Frame frame = {prog, map, {&arr, &result.arr}};
        video.grid(frame, {w, h}, sync);
    }
}

template<typename T>
void mcf::Mat<T>::transform(const Mat<T>& X, const std::function<T(const T&, const T&)>& f, Mat<T>& result, TRANSPOSE option) const{
    if(option == NONE){
        requireMatrixShape(X, h, w, "transform");
        requireMatrixShape(result, h, w, "transform", true);

		#ifdef MATRIXCF_USE_OPENMP
		#pragma omp parallel for
		#endif
        for(size_t i = 0; total_size > i; i++) result.arr[i] = f(arr[i], X.arr[i]);

    }else if(option == FIRST){
        requireMatrixShape(X, w, h, "transform");
        requireMatrixShape(result, w, h, "transform", true);

		#ifdef MATRIXCF_USE_OPENMP
		#pragma omp parallel for collapse(2)
		#endif
        for(size_t i = 0; w > i; i++){
            for(size_t j = 0; h > j; j++) result[i][j] = f(getE(j, i), X.getE(i, j));
        }

    }else if(option == SECOND){
        requireMatrixShape(*this, X.w, X.h, "transform");
        requireMatrixShape(result, X.w, X.h, "transform", true);

		#ifdef MATRIXCF_USE_OPENMP
		#pragma omp parallel for collapse(2)
		#endif
        for(size_t i = 0; X.w > i; i++){
            for(size_t j = 0; X.h > j; j++) result[i][j] = f(getE(i, j), X.getE(j, i));
        }
    }else{
        requireMatrixShape(X, h, w, "transform");
        requireMatrixShape(result, w, h, "transform", true);

		#ifdef MATRIXCF_USE_OPENMP
		#pragma omp parallel for collapse(2)
		#endif
        for(size_t i = 0; w > i; i++){
            for(size_t j = 0; h > j; j++) result[i][j] = f(getE(j, i), X.getE(j, i));
        }
    }
}
template<typename T>
void mcf::Mat<T>::transform(const Mat<T>& X, const std::string& body, Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const{
    if(option == NONE){
        requireMatrixShape(X, h, w, "transform");
        requireMatrixShape(result, h, w, "transform", true);

        std::string type = getTypeName();

        ecl::Program prog = "__kernel void transform";
        prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
        prog += "{\n";
        prog += "size_t index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
        prog += type + " v1 = a[index];\n";
        prog += type + " v2 = b[index];\n";
        prog += type + " ret;\n";
        prog += body + "\n";
        prog += "result[index] = ret;\n";
        prog += "}";

        ecl::Kernel transform = "transform";

        ecl::Frame frame = {prog, transform, {&arr, &X.arr, &result.arr}};
        video.grid(frame, {h, w}, sync);

    }else if(option == FIRST){
        requireMatrixShape(X, w, h, "transform");
        requireMatrixShape(result, w, h, "transform", true);

        std::string type = getTypeName();

        ecl::Program prog = "__kernel void transform";
        prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
        prog += "{\n";
        prog += "size_t result_index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
        prog += "size_t index = get_global_id(1) * get_global_size(0) + get_global_id(0);\n";
        prog += type + " v1 = a[index];\n";
        prog += type + " v2 = b[result_index];\n";
        prog += type + " ret;\n";
        prog += body + "\n";
        prog += "result[result_index] = ret;\n";
        prog += "}";

        ecl::Kernel transform = "transform";

        ecl::Frame frame = {prog, transform, {&arr, &X.arr, &result.arr}};
        video.grid(frame, {w, h}, sync);

    }else if(option == SECOND){
        requireMatrixShape(*this, X.w, X.h, "transform");
        requireMatrixShape(result, X.w, X.h, "transform", true);

        std::string type = getTypeName();

        ecl::Program prog = "__kernel void transform";
        prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
        prog += "{\n";
        prog += "size_t result_index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
        prog += "size_t index = get_global_id(1) * get_global_size(0) + get_global_id(0);\n";
        prog += type + " v1 = a[result_index];\n";
        prog += type + " v2 = b[index];\n";
        prog += type + " ret;\n";
        prog += body + "\n";
        prog += "result[result_index] = ret;\n";
        prog += "}";

        ecl::Kernel transform = "transform";

        ecl::Frame frame = {prog, transform, {&arr, &X.arr, &result.arr}};
        video.grid(frame, {X.w, X.h}, sync);
    }else{
        requireMatrixShape(X, h, w, "transform");
        requireMatrixShape(result, w, h, "transform", true);

        std::string type = getTypeName();

        ecl::Program prog = "__kernel void transform";
        prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
        prog += "{\n";
        prog += "size_t result_index = get_global_id(0) * get_global_size(1) + get_global_id(1);\n";
        prog += "size_t index = get_global_id(1) * get_global_size(0) + get_global_id(0);\n";
        prog += type + " v1 = a[index];\n";
        prog += type + " v2 = b[index];\n";
        prog += type + " ret;\n";
        prog += body + "\n";
        prog += "result[result_index] = ret;\n";
        prog += "}";

        ecl::Kernel transform = "transform";

        ecl::Frame frame = {prog, transform, {&arr, &X.arr, &result.arr}};
        video.grid(frame, {w, h}, sync);
    }
}

// methods (immutable)
template<typename T>
void mcf::Mat<T>::transpose(Mat<T>& result) const{
    map([](const T& v){
        return v;
    }, result, FIRST);
}
template<typename T>
void mcf::Mat<T>::transpose(Mat<T>& result, ecl::Computer& video, ecl::EXEC sync) const{
    map("ret = v;", result, video, FIRST, sync);
}

template<typename T>
void mcf::Mat<T>::reduce(Mat<T>& result, REDUCE option, TRANSPOSE transpose_option) const{
    if(transpose_option == NONE){

        if(option == FULL){
            requireMatrixShape(result, 1, 1, "reduce", true);
            result.zeros();

            for(size_t i = 0; total_size > i; i++) result[0][0] += arr[i];
        }else if(option == ROWS){
            requireMatrixShape(result, 1, w, "reduce", true);
            result.zeros();

            // #pragma omp parallel for
            for(size_t j = 0; w > j; j++){
                for(size_t i = 0; h > i; i++) result[0][j] += getE(i, j);
            }
        } else if(option == COLUMNS){
            requireMatrixShape(result, h, 1, "reduce", true);
            result.zeros();

            // #pragma omp parallel for
            for(size_t i = 0; h > i; i++){
                for(size_t j = 0; w > j; j++) result[i][0] += getE(i, j);
            }
        }
    }else{
            if(option == FULL){
            requireMatrixShape(result, 1, 1, "reduce", true);
            result.zeros();

            // #pragma omp parallel for
            for(size_t j = 0; h > j; j++){
                for(size_t i = 0; w > i; i++) result[0][0] += getE(j, i);
            }
        }else if(option == ROWS){
            requireMatrixShape(result, 1, h, "reduce", true);
            result.zeros();

            // #pragma omp parallel for
            for(size_t j = 0; h > j; j++){
                for(size_t i = 0; w > i; i++) result[0][j] += getE(j, i);
            }
        } else if(option == COLUMNS){
            requireMatrixShape(result, w, 1, "reduce", true);
            result.zeros();

            // #pragma omp parallel for
            for(size_t i = 0; w > i; i++){
                for(size_t j = 0; h > j; j++) result[i][0] += getE(j, i);
            }
        }
    }
}
template<typename T>
void mcf::Mat<T>::reduce(Mat<T>& result, ecl::Computer& video, REDUCE option, TRANSPOSE transpose_option, ecl::EXEC sync) const{
    if(transpose_option == NONE){
        if(option == FULL){
            throw std::runtime_error("full reduce on ecl::Computer temporary unavailable");
        }else if(option == ROWS){
            requireMatrixShape(result, 1, w, "reduce", true);
            result.zeros(video);

            std::string type = getTypeName();

            ecl::Program prog = "__kernel void reduce";
            prog += "(__global " + type + "* a, __global " + type + "* result)";
            prog += "{\n";
            prog += "size_t j = get_global_id(0);\n";
            prog += "size_t w = get_global_size(0);\n";
            prog += type + " sum = 0;\n";
            prog += "for(size_t i = 0; i < " + std::to_string(h) + "; i++){\n";
            prog += "sum += a[i * w + j];\n";
            prog += "}\n";
            prog += "result[j] = sum;\n";
            prog += "}";

            ecl::Kernel reduce = "reduce";

            ecl::Frame frame = {prog, reduce, {&arr, &result.arr}};
            video.grid(frame, {w}, sync);

        } else if(option == COLUMNS){
            requireMatrixShape(result, h, 1, "reduce", true);
            result.zeros(video);

            std::string type = getTypeName();

            ecl::Program prog = "__kernel void reduce";
            prog += "(__global " + type + "* a, __global " + type + "* result)";
            prog += "{\n";
            prog += "size_t i = get_global_id(0);";
            prog += "size_t h = get_global_size(0);";
            prog += "size_t w = " + std::to_string(w) + ";";
            prog += "size_t iw = i * w;\n";
            prog += type + " sum = 0;\n";
            prog += "for(size_t j = 0; j < w; j++){\n";
            prog += "sum += a[iw + j];";
            prog += "}\n";
            prog += "result[i] = sum;\n";
            prog += "}";

            ecl::Kernel reduce = "reduce";

            ecl::Frame frame = {prog, reduce, {&arr, &result.arr}};
            video.grid(frame, {h}, sync);
        }
    }else{
        if(option == FULL){
            throw std::runtime_error("full reduce on ecl::Computer temporary unavailable");
        }else if(option == ROWS){
            requireMatrixShape(result, 1, h, "reduce", true);
            result.zeros(video);

            std::string type = getTypeName();

            ecl::Program prog = "__kernel void reduce";
            prog += "(__global " + type + "* a, __global " + type + "* result)";
            prog += "{\n";
            prog += "size_t j = get_global_id(0);\n";
            prog += "size_t w = " + std::to_string(w) + ";\n";
            prog += type + " sum = 0;\n";
            prog += "for(size_t i = 0; i < w; i++){\n";
            prog += "sum += a[j * w + i];\n";
            prog += "}\n";
            prog += "result[j] = sum;\n";
            prog += "}";

            ecl::Kernel reduce = "reduce";

            ecl::Frame frame = {prog, reduce, {&arr, &result.arr}};
            video.grid(frame, {h}, sync);

        } else if(option == COLUMNS){
            requireMatrixShape(result, w, 1, "reduce", true);
            result.zeros(video);

            std::string type = getTypeName();

            ecl::Program prog = "__kernel void reduce";
            prog += "(__global " + type + "* a, __global " + type + "* result)";
            prog += "{\n";
            prog += "size_t i = get_global_id(0);";
            prog += "size_t h = " + std::to_string(h) + ";";
            prog += "size_t w = get_global_size(0);";
            prog += type + " sum = 0;\n";
            prog += "for(size_t j = 0; j < h; j++){\n";
            prog += "sum += a[j * w + i];";
            prog += "}\n";
            prog += "result[i] = sum;\n";
            prog += "}";

            ecl::Kernel reduce = "reduce";

            ecl::Frame frame = {prog, reduce, {&arr, &result.arr}};
            video.grid(frame, {w}, sync);
        }
    }
}

template<typename T>
T mcf::Mat<T>::reduce() const{
    T result = 0;

    for(size_t i = 0; i < total_size; i++) result += arr[i];

    return result;
}
template<typename T>
T mcf::Mat<T>::mreduce(const std::function<T(const T&)>& f) const {
	T result = 0;

	for (size_t i = 0; i < total_size; i++) result += f(arr[i]);

	return result;
}

template<typename T>
void mcf::Mat<T>::add(const Mat<T>& X, Mat<T>& result, TRANSPOSE option) const{
    transform(X, [](const T& v1, const T& v2){
        return v1 + v2;
    }, result, option);
}
template<typename T>
void mcf::Mat<T>::add(const Mat<T>& X, Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const{
    transform(X, "ret = v1 + v2;", result, video, option, sync);
}

template<typename T>
void mcf::Mat<T>::sub(const Mat<T>& X, Mat<T>& result, TRANSPOSE option) const{
    transform(X, [](const T& v1, const T& v2){
        return v1 - v2;
    }, result, option);
}
template<typename T>
void mcf::Mat<T>::sub(const Mat<T>& X, Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const{
    transform(X, "ret = v1 - v2;", result, video, option, sync);
}

template<typename T>
void mcf::Mat<T>::hadamard(const Mat<T>& X, Mat<T>& result, TRANSPOSE option) const{
    transform(X, [](const T& v1, const T& v2){
        return v1 * v2;
    }, result, option);
}
template<typename T>
void mcf::Mat<T>::hadamard(const Mat<T>& X, Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const{
    transform(X, "ret = v1 * v2;", result, video, option, sync);
}

template<typename T>
void mcf::Mat<T>::mul(const Mat<T>& X, Mat<T>& result, TRANSPOSE option) const{
    size_t first_h = h;
    size_t first_w = w;
    size_t second_h = X.h;
    size_t second_w = X.w;

    if (option == FIRST){
        first_h = w;
        first_w = h;
    }else if(option == SECOND){
        second_h = X.w;
        second_w = X.h;
    }else if(option == BOTH){
        first_h = w;
        first_w = h;
        second_h = X.w;
        second_w = X.h;
    }

    requireMatrixShape(result, first_h, second_w, "mul", true);
    requireMatrixH(first_w, second_h, "mul");

    result.zeros();

    if(option == NONE){    
        // #pragma omp parallel for
        for(size_t i = 0; first_h > i; i++)
            for(size_t k = 0; first_w > k; k++)
                for(size_t j = 0; second_w > j; j++)
                    result[i][j] += getE(i, k) * X.getE(k, j);

    }else if(option == FIRST){
        // #pragma omp parallel for
        for(size_t i = 0; first_h > i; i++)
            for(size_t k = 0; first_w > k; k++) 
                for(size_t j = 0; second_w > j; j++)
                    result[i][j] += getE(k, i) * X.getE(k, j);

    }else if(option == SECOND){    
        // #pragma omp parallel for
        for(size_t i = 0; first_h > i; i++)
            for(size_t j = 0; second_w > j; j++)
                for(size_t k = 0; first_w > k; k++)
                    result[i][j] += getE(i, k) * X.getE(j, k);
    }else{
        // #pragma omp parallel for
        for(size_t i = 0; first_h > i; i++)
            for(size_t j = 0; second_w > j; j++)
                for(size_t k = 0; first_w > k; k++)
                    result[i][j] += getE(k, i) * X.getE(j, k);
    }
}
template<typename T>
void mcf::Mat<T>::mul(const Mat<T>& X, Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const{
    std::string type = getTypeName();

    size_t first_h = h;
    size_t first_w = w;
    size_t second_h = X.h;
    size_t second_w = X.w;

    std::string mul_optimized = "a[i * " + std::to_string(w) + " + k] * b[k * " + std::to_string(X.w) + " + j];";

    if (option == FIRST){
        first_h = w;
        first_w = h;
        mul_optimized = "a[k * " + std::to_string(w) + " + i] * b[k * " + std::to_string(X.w) + " + j];";
    }else if(option == SECOND){
        second_h = X.w;
        second_w = X.h;
        mul_optimized = "a[i * " + std::to_string(w) + " + k] * b[j * " + std::to_string(X.w) + " + k];";
    }else if(option == BOTH){
        first_h = w;
        first_w = h;
        second_h = X.w;
        second_w = X.h;
        mul_optimized = "a[k * " + std::to_string(w) + " + i] * b[j * " + std::to_string(X.w) + " + k];";
    }

    requireMatrixShape(result, first_h, second_w, "mul", true);
    requireMatrixH(first_w, second_h, "mul");


    ecl::Program prog = "__kernel void mul";
    prog += "(__global " + type +  "* a, __global " + type + "* b, __global " + type + "* result)";
    prog += "{\n";
    prog += "size_t i = get_global_id(0);\n";
    prog += "size_t j = get_global_id(1);\n";
    prog += "size_t w = get_global_size(1);\n";
    prog += type + " sum = 0;\n";
    prog += "for(size_t k = 0; k < " + std::to_string(first_w) + "; k++) ";
    prog += "sum += " + mul_optimized + "\n";
    prog += "result[i * w + j] = sum;\n";
    prog += "}";

    ecl::Kernel mul = "mul";

    ecl::Frame frame = {prog, mul, {&arr, &X.arr, &result.arr}};
    video.grid(frame, {first_h, second_w}, sync);
}

template<typename T>
void mcf::Mat<T>::mul(const T& value, Mat<T>& result, TRANSPOSE option) const{
    map([&](const T& v){
        return v * value;
    }, result, option);
}
template<typename T>
void mcf::Mat<T>::mul(const T& value, Mat<T>& result, ecl::Computer& video, TRANSPOSE option, ecl::EXEC sync) const{
    std::string val = std::to_string(value);
    map("ret = v * " + val + ";", result, video, option, sync);
}

template<typename T>
void mcf::Mat<T>::hsplit(Mat<T>& A, Mat<T>& B) const{
    requireMatrixH(A.h, B.h, "hsplit");
    requireMatrixShape(*this, A.h, A.w + B.w, "hsplit", true);

    // #pragma omp parallel for collapse(2)
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++){
            if(A.w > j) A[i][j] = getE(i, j);
            else B[i][j - A.w] = getE(i, j);
        }
    }
}
template<typename T>
void mcf::Mat<T>::hsplit(Mat<T>& A, Mat<T>& B, ecl::Computer& video, ecl::EXEC sync) const{
    std::string type = getTypeName();

    requireMatrixH(A.h, B.h, "hsplit");
    requireMatrixShape(*this, A.h, A.w + B.w, "hsplit", true);

    ecl::Program prog = "__kernel void hsplit";
    prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
    prog += "{\n";
    prog += "size_t i = get_global_id(0);\n";
    prog += "size_t j = get_global_id(1);\n";
    prog += "size_t w = get_global_size(1);\n";
    prog += "size_t a_w = " + std::to_string(A.w) + ";\n";
    prog += "size_t b_w = " + std::to_string(B.w) + ";\n";
    prog += "if(j < a_w) a[i * a_w + j] = result[i * w + j];\n";
    prog += "else b[i * b_w + j - a_w] = result[i * w + j];\n";
    prog += "}";

    ecl::Kernel hsplit = "hsplit";

    ecl::Frame frame = {prog, hsplit, {&A.arr, &B.arr, &arr}};
    video.grid(frame, {h, w}, sync);
}

template<typename T>
void mcf::Mat<T>::vsplit(Mat<T>& A, Mat<T>& B) const{
    requireMatrixH(A.w, B.w, "vsplit");
    requireMatrixShape(*this, A.h + B.h, A.w, "vsplit", true);

    // #pragma omp parallel for collapse(2)
    for(size_t i = 0; h > i; i++){
        for(size_t j = 0; w > j; j++){
            if(A.h > i) A[i][j] = getE(i, j);
            else B[i - A.h][j] = getE(i, j);
        }
    }
}
template<typename T>
void mcf::Mat<T>::vsplit(Mat<T>& A, Mat<T>& B, ecl::Computer& video, ecl::EXEC sync) const{
    std::string type = getTypeName();

    requireMatrixH(A.w, B.w, "vsplit");
    requireMatrixShape(*this, A.h + B.h, A.w, "vsplit", true);

    ecl::Program prog = "__kernel void vsplit";
    prog += "(__global " + type + "* a, __global " + type + "* b, __global " + type + "* result)";
    prog += "{\n";
    prog += "size_t i = get_global_id(0);\n";
    prog += "size_t j = get_global_id(1);\n";
    prog += "size_t w = get_global_size(1);\n";
    prog += "size_t a_h = " + std::to_string(A.h) + ";\n";
    prog += "size_t a_w = " + std::to_string(A.w) + ";\n";
    prog += "size_t b_w = " + std::to_string(B.w) + ";\n";
    prog += "if(i < a_h) a[i * a_w + j] = result[i * w + j];\n";
    prog += "else b[(i - a_h) * b_w + j] = result[i * w + j];\n";
    prog += "}";

    ecl::Kernel vsplit = "vsplit";

    ecl::Frame frame = {prog, vsplit, {&A.arr, &B.arr, &arr}};
    video.grid(frame, {h, w}, sync);
}


template<typename T>
mcf::Mat<T>::~Mat(){
    clear();
}