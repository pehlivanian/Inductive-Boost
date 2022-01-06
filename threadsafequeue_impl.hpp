#ifndef __THREADSAFEQUEUE_IMPL_HPP__
#define __THREADSAFEQUEUE_IMPL_HPP__

template<typename T>
ThreadsafeQueue<T>::~ThreadsafeQueue() {
  invalidate();
}

template<typename T>
bool
ThreadsafeQueue<T>::tryPop(T& out) {
  std::lock_guard<std::mutex> lock{m_mutex};
  if(m_queue.empty() || !m_valid)
    {
      return false;
    }
  out = std::move(m_queue.front());
  m_queue.pop();
  return true;
}

template<typename T>
bool
ThreadsafeQueue<T>::waitPop(T& out) {
  std::unique_lock<std::mutex> lock{m_mutex};
  m_condition.wait(lock, [this]()
		   {
		     return !m_queue.empty() || !m_valid;
		   });
  /*
   * Using the condition in the predicate ensures that spurious wakeups with a valid
   * but empty queue will not proceed, so only need to check for validity before proceeding.
   */
  if(!m_valid)
    {
      return false;
    }
    out = std::move(m_queue.front());
    m_queue.pop();
    return true;
}

template<typename T>
void
ThreadsafeQueue<T>::push(T value) {
  std::lock_guard<std::mutex> lock{m_mutex};
  m_queue.push(std::move(value));
  m_condition.notify_one();
}

template<typename T>
bool
ThreadsafeQueue<T>::size(void) const {
  std::lock_guard<std::mutex> lock{m_mutex};
  return m_queue.size();
}

template<typename T>
bool
ThreadsafeQueue<T>::empty(void) const {
  std::lock_guard<std::mutex> lock{m_mutex};
  return m_queue.empty();
}

template<typename T>
void
ThreadsafeQueue<T>::clear(void) {
  std::lock_guard<std::mutex> lock{m_mutex};
  while(!m_queue.empty())
    {
      m_queue.pop();
    }
  m_condition.notify_all();
}

template<typename T>
void
ThreadsafeQueue<T>::invalidate(void) {
  std::lock_guard<std::mutex> lock{m_mutex};
  m_valid = false;
  m_condition.notify_all();
}

template<typename T>
bool
ThreadsafeQueue<T>::isValid(void) const {
  std::lock_guard<std::mutex> lock{m_mutex};
  return m_valid;
}

#endif
