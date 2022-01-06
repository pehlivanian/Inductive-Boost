#ifndef __THREADSAFEQUEUE_HPP__
#define __THREADSAFEQUEUE_HPP__

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>


//
// Modified version of
// http://roar11.com/2016/01/a-platform-independent-thread-pool-using-c14/
//
 
template<typename T>
class ThreadsafeQueue {

public:
  ~ThreadsafeQueue();
  
  bool tryPop(T&);
  bool waitPop(T&);
  void push(T);
  bool empty() const;
  bool size() const;
  void clear();
  void invalidate();
  bool isValid() const;

private:
  std::atomic_bool m_valid{true};
  mutable std::mutex m_mutex;
  std::queue<T> m_queue;
  std::condition_variable m_condition;

};

#include "threadsafequeue_impl.hpp"

#endif
