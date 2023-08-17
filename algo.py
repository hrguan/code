class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res = 0
        left = 0
        right = len(height)-1
        while left < right:
            l = height[left]
            r = height[right]
            temp = min(l, r) * (right-left)
            if temp > res:
                res = temp
            if l < r:
                left += 1
            else:
                right -= 1
        return res
    
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height)-1
        left_max = 0
        right_max = 0
        res = 0
        while left < right:
            if height[left] <= height[right]:
                #go left
                if height[left] > left_max:
                    left_max = height[left]
                else:
                    res += (left_max-height[left])
                left += 1
            else:
                #go right
                if height[right] > right_max:
                    right_max = height[right]
                else:
                    res += (right_max-height[right])
                right -= 1


        return res
    
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # space
        # time
        if len(nums) < 3:
            return []
        res = set()
        nums.sort()
        for i in range(0, len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                if nums[i]+nums[j]+nums[k]==0:
                    res.add((nums[i], nums[j], nums[k]))
                    j += 1
                    k -= 1
                    
                elif nums[i]+nums[j]+nums[k] < 0:
                    j+=1
                else:
                    k-=1

        return res
    
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # space
        # time
        res = float("inf")
        nums.sort()
        if len(nums) < 3:
            return 0
        for i in range(0, len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                temp = nums[i]+nums[j]+nums[k]
                if temp == target:
                    return target
                elif temp < target:
                    j+=1      
                else:
                    k-=1
                curr_diff = abs(temp-target)
                prev_diff = abs(res-target)

                if curr_diff < prev_diff:
                    res = temp
        return res
    
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = set()
        if len(nums) < 4:
            return []
        nums.sort()

        for a in range(0, len(nums)-3):
            for b in range(a+1, len(nums)-2):
                c = b+1
                d = len(nums)-1     
                while c < d:
                    temp = nums[a]+nums[b]+nums[c]+nums[d]
                    if temp == target:
                        res.add((nums[a], nums[b], nums[c], nums[d]))
                        c += 1
                        d -= 1
                    elif temp < target:
                        c += 1
                    else:
                        d -= 1
        return res
    
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero = 0
        one = 0
        two = len(nums)-1

        while one <= two:
            if nums[one] == 0:
                nums[one], nums[zero] = nums[zero], nums[one]
                zero += 1
                one += 1
            elif nums[one] == 1:
                one += 1
            else:
                nums[one], nums[two] = nums[two], nums[one]
                two -= 1

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left = 0
        right = len(s)-1
        
        while left <= right:
            l = ""
            r = ""
            while left <= len(s)-1 and not s[left].isalnum():
                left += 1
            if left <= len(s)-1:
                l = s[left].lower()
            while right >= 0 and not s[right].isalnum():
                right -= 1
            if right >= 0:
                r = s[right].lower()
            if l != r:
                return False
            left += 1
            right -=1
            

        return True
    
    def validPalindrome(self, s):
        left = 0
        right = len(s)-1

        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.check(s, left+1, right) or self.check(s, left, right-1)
        return True
    
    def check(self, s, left, right):
        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False
        return True
    
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        words = s.split()
        res = ""
        p = len(words)
        while p:
            if p == len(words):
                res += words[p-1]
            else:
                res += " " + words[p-1]
            p -= 1

        return res

    def reverseWords(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        print(s)
        
        start = 0
        end = 0
        while end <= len(s)-1:
            while end <= len(s)-1 and s[end] != " ":
                end += 1
            e = end -1
            while start <= e:
                s[start], s[e] = s[e], s[start]
                start += 1
                e -= 1
            start = end+1
            end += 1
        
        ee = len(s)-1
        while start <= ee:
            s[start], s[ee] = s[ee], s[start]
            start += 1
            ee -= 1

    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
        start = 0
        end = 0
        while end <= len(s)-1:
            while end <= len(s)-1 and s[end] != " ":
                end += 1
            e = end-1
            while start <= e:
                s[start], s[e] = s[e], s[start]
                start += 1
                e -= 1
            start = end+1
            end += 1
        
        e = len(s)-1
        while start <= e:
            s[start], s[e] = s[e], s[start]
            start += 1
            e -= 1
        
        return "".join(s)
    
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        seen = set()
        while head:
            if head in seen:
                return head
            seen.add(head)
            head = head.next
        return None
    
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False
    
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # left = 0
        # right = len(nums)-1

        # if nums[left] == nums[right]:
        #     return nums[left]
        # while left < right:
        #     middle = left+1
        #     while middle < right:
        #         if nums[middle] == nums[right] or nums[middle] == nums[left]:
        #             return nums[middle]
        #         middle += 1
        #     left += 1
        #     right -= 1

        s = nums[0]
        f = nums[0]

        while True:
            s = nums[s]
            f = nums[nums[f]]
            if s == f:
                break
        
        s = nums[0]

        while s != f:
            s = nums[s]
            f = nums[f]

        return s

    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        intervals.sort(key=lambda x:x[0])
        print(intervals)
        need = Queue.PriorityQueue()
        need.put(intervals[0][1])
        for i in range(1, len(intervals)):
            start = intervals[i][0]
            end = intervals[i][1]
            curr_end = need.get()
            if start < curr_end:
                need.put(curr_end)
            need.put(end)
        return need.qsize()

    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = head
        fast = head
        prev = None

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = slow
        slow = slow.next
        prev.next = None

        while slow:
            n = slow.next
            slow.next = prev
            prev = slow
            slow = n
        
        slow = prev
        fast = head

        while slow:
            if fast.val != slow.val :
                return False
            fast = fast.next
            slow = slow.next
        return True
    
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        last = m+n-1
        a = m-1
        b = n-1

        while b >= 0:
            if a >= 0 and nums1[a] > nums2[b]:
                nums1[last] = nums1[a]
                a -= 1
                
            else:
                nums1[last] = nums2[b]
                b -= 1
                
            last -= 1

    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        before_head = ListNode(0)
        before = before_head
        after_head = ListNode(0)
        after = after_head

        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        after.next = None
        before.next = after_head.next
        return before_head.next
    
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr = nums[0]
        idx = 1
        for i in range(1, len(nums)):
            if nums[i] == curr:
                continue
            nums[idx] = nums[i]
            curr = nums[i]
            idx +=1
        return idx
    
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen = set()
        while True:
            if n == 1:
                return True
            elif n in seen:
                return False
            else:
                seen.add(n)
                temp = 0
                for i in str(n):
                    temp += int(math.pow(int(i), 2))
                n = temp
    
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        #find peak
        for i in range(len(nums)-1, 0, -1):
            if nums[i - 1] < nums[i]:
                nums[i:] = sorted(nums[i:])

                pre_idx = i - 1
            
                for i in range(i, len(nums)):
                    if nums[pre_idx] < nums[i]:
                        nums[pre_idx], nums[i] = nums[i], nums[pre_idx]
                        return nums
        
        return nums.reverse()
    
    def circularArrayLoop(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        visit = set()
        for i in range(len(nums)):
            cycle = set()
            if i not in visit:
                while True:
                    if i in cycle:
                        return True
                    if i in visit:
                        break
                    visit.add(i)
                    cycle.add(i)
                    prev = i
                    i = (i+nums[i]) % len(nums)
                    if prev == i or ((nums[prev]>0) != (nums[i]>0)):
                        break
        return False
    
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) > len(s2):
            return False
        
        cnt1 = [0]*26
        cnt2 = [0]*26

        for i in range(len(s1)):
            cnt1[ord(s1[i])-ord('a')] += 1

        for i in range(len(s2)):
            cnt2[ord(s2[i])-ord('a')] += 1
            if i >= len(s1):
                cnt2[ord(s2[i-len(s1)])-ord('a')] -= 1

            if cnt1 == cnt2:
                return True

        return False

    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        res = []
        d = dict()
        for idx, char in enumerate(s):
            d[char] = idx
        
        start = 0
        end = 0
        for idx, char in enumerate(s):
            end = max(end, d[char])
            if end == idx:
                res.append(end-start+1)
                start = end+1
        
        return res
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0 
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res = max(res, prices[i]-buy)
        return res
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res += (prices[i] - buy)
                buy = prices[i]

        return res
    
    def totalFruit(self, fruits):
        """
        :type fruits: List[int]
        :rtype: int
        """
        res = 0
        d = collections.defaultdict(int)

        l = 0
        for r, fruit in enumerate(fruits):
            d[fruit] += 1
            while len(d) > 2:
                d[fruits[l]] -= 1
                if d[fruits[l]] == 0:
                    del d[fruits[l]]
                l += 1
            res = max(res, r-l+1)

        return res
    
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left = 0
        res = 0
        right = 0
        seen = dict()
        while right < len(s):
            if s[right] in seen:
                left = max(left, seen[s[right]]+1)
            res = max(res, right-left+1)
            seen[s[right]] = right
            right += 1
        return res
    
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res, d = [], {}
        for i in range(len(s)):
            if s[i:i+10] not in d: 
                d[s[i :i+10]] = 0
            else:
                if s[i:i+10] not in res: 
                    res.append(s[i:i+10])
        
        return res
    
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        left = 0
        right = len(arr) - 1

        while right-left >= k:
            if abs(x-arr[left]) <= abs(x-arr[right]):  
                right -= 1
            else:
                left += 1
        return arr[left:right+1]
    
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        if sum(nums) < target:
            return 0
        elif sum(nums) == target:
            return len(nums)

        res = float("inf")
        left = 0
        temp = nums[left]
        if temp >= target:
            res = 1
        for right in range(1, len(nums)):   
            temp += nums[right]
            if temp >= target:
                res = min(res, right-left+1)
                while temp >= target:
                    temp -= nums[left]
                    left += 1
                    if temp >= target:
                        res = min(res, right-left+1)
        return res

    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        q = collections.deque()
        res = []
        for i in range(len(nums)):
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if q[0] == i-k:
                q.popleft()
            if i >= k-1:
                res.append(nums[q[0]])
        return res
    
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if len(t) > len(s):
            return ""
        
        left = 0
        start = 0
        end = 0
        missing = len(t)
        need = collections.Counter(t)
        for right, char in enumerate(s, 1):
            if need[char] > 0:
                missing -= 1
            need[char] -= 1
            if missing == 0:
                while left < right and need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                need[s[left]] += 1
                missing += 1
                if end == 0 or right -left < end-start:
                    start = left
                    end = right
                left+=1
        return s[start:end]
    
    def findLength(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        n2 = "".join([chr(x) for x in nums2])
        curr = ""
        res = 0
        for num in nums1:
            curr += chr(num)
            if curr in n2:
                res = max(res, len(curr))
            else:
                curr = curr[1:]
        return res

    def countSubarrays(self, nums, minK, maxK):
        """
        :type nums: List[int]
        :type minK: int
        :type maxK: int
        :rtype: int
        """
        currMin = -1
        currMax = -1
        currOver = -1
        res = 0
        for i in range(len(nums)):
            if nums[i] == minK:
                currMin = i
            if nums[i] == maxK:
                currMax = i
            if nums[i] > maxK or nums[i] < minK:
                currOver = i
            if currMin != -1 and currMax != -1:
                start = min(currMin, currMax)
                if currOver < start:
                    res += start - currOver
        return res
    
    def maxTurbulenceSize(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        res = 0
        temp = 0
        for i in range(len(arr)):
            if i >= 2 and (arr[i-2] > arr[i-1] < arr[i] or arr[i-2] < arr[i-1] > arr[i]):
                    temp += 1
            elif i >= 1:
                if arr[i] != arr[i-1]:
                    temp = 2
            else:
                temp = 1
            res = max(res, temp)
        return res

    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        if not intervals:
            return True
        intervals.sort(key=lambda x:x[0])
        end = intervals[0][1]
        q = Queue.PriorityQueue()
        q.put(end)
        for i in range(1, len(intervals)):
            prev_end = q.get()
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if curr_start < prev_end:
                q.put(prev_end)
            q.put(curr_end)
        return True if q.qsize() == 1 else False
    
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key=lambda x:x[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            end = res[-1][1]
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if curr_start <= end:
                res[-1][1] = max(curr_end, end)
            else:
                res.append(intervals[i])
                end = curr_end
        return res
    
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        res = []
        for i in range(len(intervals)):
            if intervals[i][1] < newInterval[0]:
                res.append(intervals[i])
            elif intervals[i][0] > newInterval[1]:
                res.append(newInterval)
                newInterval = intervals[i]
            elif intervals[i][1] >= newInterval[0]:
                newInterval[0] = min(intervals[i][0], newInterval[0])
                newInterval[1] = max(intervals[i][1], newInterval[1])  
        res.append(newInterval)
        return res
    
    def intervalIntersection(self, firstList, secondList):
        """
        :type firstList: List[List[int]]
        :type secondList: List[List[int]]
        :rtype: List[List[int]]
        """
        i = 0
        j = 0
        res = []
        while i < len(firstList) and j < len(secondList):
            first_start = firstList[i][0]
            first_end = firstList[i][1]
            second_start = secondList[j][0]
            second_end = secondList[j][1]
            if first_start <= second_end and second_start <= first_end:
                res.append([max(first_start, second_start), min(first_end, second_end)])
            if first_end <= second_end:
                i += 1
            else:
                j += 1
        return res
    
    def employeeFreeTime(self, schedule):
        """
        :type schedule: [[Interval]]
        :rtype: [Interval]
        """
        intervals = []
        for employee in schedule:
            for interval in employee:
                intervals.append(interval)
        intervals.sort(key = lambda interval: interval.start)

        freeTime = []
        end = intervals[0].end
        for i in range(1, len(intervals)):
            start = intervals[i].start
            if start > end:
                freeTime.append(Interval(start = end, end = start))
                end = intervals[i].end
            else:
                end = max(intervals[i].end, end)
        return freeTime

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head
        while curr:
            n = curr.next
            curr.next = prev
            prev = curr
            curr = n
        return prev
    
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        seen = set()
        while headA:
            seen.add(headA)
            headA = headA.next
        while headB:
            if headB in seen:
                return headB
            headB = headB.next
        return None
    
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        first = head
        last = head
        for i in range(1, k):
            first = first.next
            
        null_checker = first 
        while null_checker.next:
            last = last.next
            null_checker = null_checker.next
        first.val, last.val = last.val, first.val
        return head

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        jump = dummy
        dummy.next = head
        left = right = head

        while True:
            count = 0
            while right and count < k:
                count += 1
                right = right.next
            
            if count == k:
                prev = right
                curr = left
                for i in range(k):
                    nxt = curr.next
                    curr.next = prev
                    prev = curr
                    curr = nxt
                jump.next = prev
                jump = left
                left = right

            else:
                return dummy.next

    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        prev = dummy
        dummy.next = head
        for i in range(left-1):
            prev = prev.next
        curr = prev.next
        tail = None

        for i in range(right-left+1):
            nxt = curr.next
            curr.next = tail
            tail = curr
            curr = nxt
        
        prev.next.next = curr
        prev.next = tail

        return dummy.next
    
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        while head and head.next:
            tmp = head.next
            head.next = tmp.next
            tmp.next = head
            p.next = tmp
            
            head = head.next
            p = tmp.next
        return dummy.next
    
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        heap = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i))
        
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, i = heapq.heappop(heap)
            curr.next = lists[i]
            curr = curr.next
            if lists[i].next:
                lists[i] = lists[i].next
                heapq.heappush(heap, (lists[i].val, i))
        return dummy.next
    
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head:
            return None
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = None
        curr = slow

        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        
        head_of_second_rev = prev
        
        first = head
        second = prev

        while second.next:
            nxt = first.next
            first.next = second
            first = nxt

            nxt = second.next
            second.next = first
            second = nxt

    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        def add(temp):
            curr = dummy
            while curr:
                if curr.val < temp.val:
                    prev = curr
                    curr = curr.next
                else:
                    break
            temp.next = curr
            prev.next = temp

        dummy = ListNode(-5001)
        while head:
            temp = head
            head = head.next
            temp.next = None
            add(temp)
        return dummy.next

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        
        arr.sort()
        dummy = ListNode(0)
        curr = dummy
        for i in range(len(arr)):
            curr.next = ListNode(arr[i])
            curr = curr.next
        return dummy.next

    def reverseEvenLengthGroups(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        group = 2
        tail = head 
        while tail and tail.next:
            cnt = 1 
            cur = tail.next
            while cur.next and cnt < group:
                cur = cur.next
                cnt += 1
            pre, cur = tail, tail.next
            if cnt % 2 == 0: 
                while cnt and cur:
                    nxt = cur.next
                    cur.next = pre
                    pre = cur
                    cur = nxt
                    cnt -= 1
                first = tail.next
                first.next = cur
                tail.next = pre
                tail = first
            else:
                while cnt and cur:
                    pre, cur = cur, cur.next
                    cnt -= 1
                tail = pre
            group += 1
        return head

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for num in nums:
            heapq.heappush(heap, -num)
        for i in range(k-1):
            heapq.heappop(heap)
        return heapq.heappop(heap)*-1
    
    def reorganizeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        d = collections.Counter(s)

        heap = []
        for key, value in d.items():
            heapq.heappush(heap, (-value, key))
        
        max_count, max_char = heapq.heappop(heap)
        if max_count*-1 > (len(s)+1)//2:
            return ""
        
        idx = 0
        res = ["" for i in range(len(s))]
        
        for i in range((max_count*-1)):
            res[idx] = max_char
            idx += 2
        while heap:
            count, char = heapq.heappop(heap)
            for i in range(count*-1):
                if idx >= len(s):
                    idx = 1
                res[idx] = char
                idx += 2
        return ''.join(res)

    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        heap = []
        for x, y in points:
            temp = math.pow(x, 2) + math.pow(y, 2)
            distance = math.sqrt(temp)

            heapq.heappush(heap, (distance, [x, y]))
        res = []
        while k > 0:
            distance, point = heapq.heappop(heap)
            res.append(point)
            k -= 1
        return res
    
    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.k = k
        self.heap = nums[:k]
        heapq.heapify(self.heap)
        for i in range(k, len(nums)):
            heapq.heappush(self.heap, nums[i])
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap)> self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        count = collections.Counter(nums)
        heap = []
        for key, value in count.items():
            heapq.heappush(heap, (value, key))
            if len(heap)>k:
                heapq.heappop(heap)
        return [k for v, k in heap]
    
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        curr = root
        while True:
            while curr:
                stack.append(curr)
                curr = curr.left
            if not stack:
                break
            node = stack.pop()
            k -= 1
            if k == 0:
                return node.val
            curr = node.right
    
    def mostBooked(self, n, meetings):
        """
        :type n: int
        :type meetings: List[List[int]]
        :rtype: int
        """
        available = [r for r in range(n)]
        meeting_rooms = []
        used = [0]*n
        meetings.sort(key=lambda x:x[0])
        for start, end in meetings:
            while meeting_rooms and meeting_rooms[0][0] <= start:
                ending, room_no = heapq.heappop(meeting_rooms)
                heapq.heappush(available, room_no)
            if available:
                room_no = heapq.heappop(available)
                heapq.heappush(meeting_rooms, (end, room_no))
                used[room_no] += 1
            else:
                ending, room_no = heapq.heappop(meeting_rooms)
                heapq.heappush(meeting_rooms, (end+ending-start, room_no))
                used[room_no] += 1


        return used.index(max(used))

    def putMarbles(self, weights, k):
        """
        :type weights: List[int]
        :type k: int
        :rtype: int
        """
        if k == 1 or len(weights) == k:
            return 0
        m = sorted(weights[i]+weights[i-1] for i in range(1, len(weights)))

        return sum(m[-k+1:]) - sum(m[:k-1])

    def findCheapestPrice(self, n, flights, src, dst, k):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type k: int
        :rtype: int
        """
        adj = [[] for i in range(n)]
        for s, d, c in flights:
            adj[s].append((d, c))
        stop = 0
        q = collections.deque()
        q.append((src, 0))
        
        cost = [ float("inf") for i in range(n)]
        while q and stop <= k:
            for i in range(len(q)):
                curr, price = q.popleft()
                for nei, c in adj[curr]:
                    if price+c >= cost[nei]:
                        continue
                    cost[nei] = price+c
                    q.append((nei, price+c))
            stop += 1

        return -1 if cost[dst] == float("inf") else cost[dst]

    def minStoneSum(self, piles, k):
        """
        :type piles: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for pile in piles:
            heapq.heappush(heap, -pile)
        for i in range(k):
            curr = heapq.heappop(heap) * -1
            remove = curr // 2
            curr -= remove
            heapq.heappush(heap, -curr)
        return sum(heap) * -1
    
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        if n == 0:
            return len(tasks)
        
        count = collections.Counter(tasks)
        freq = count.values()
        max_count = max(freq)
        last = freq.count(max_count)

        ans = (max_count-1) * (n+1) + last
        print(ans, len(tasks))
        return max(ans, len(tasks))
    
        # if n == 0:
        #     return len(tasks)
        # counter = collections.Counter(tasks)
        # res = 0
        # n += 1
        # while counter:
        #     ready = counter.most_common(n)
        #     res += len(ready)
        #     for k, v in ready:
        #         if counter[k] > 1:
        #             counter[k] -= 1
        #         else:
        #             del counter[k]
        #     if counter:
        #         res += (n-len(ready))
        # return res

    def __init__(self):
        self.low = []
        self.high = []
    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        if len(self.low) == len(self.high):
            heapq.heappush(self.low, -num)
            n = heapq.heappop(self.low)
            heapq.heappush(self.high, -n)

        else:
            heapq.heappush(self.high, num)
            n = heapq.heappop(self.high)
            heapq.heappush(self.low, -n)
    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.low) == len(self.high):
            low = self.low[0] * -1
            ans = (self.high[0] + low )/2.0
            return ans
        else:
            return float(self.high[0])
        
    def medianSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[float]
        """
        from sortedcontainers import SortedList
        res = []
        sl = SortedList()
        for i in range(len(nums)):
            sl.add(nums[i])
            if i < k-1:
                continue
            if k % 2 != 0:
                res.append(sl[k//2]*1.0)
            else:
                s = (sl[k//2] + sl[k//2-1]) / 2.0
                res.append(s)
            sl.remove(nums[i-k+1])
        return res

    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        l = matrix[0][0]
        r = matrix[-1][-1]

        def less_k(m):
            res = 0
            for i in range(len(matrix)):
                cnt = bisect_right(matrix[i], m)
                res += cnt
            return res

        while l < r:
            mid = (l+r) // 2
            if less_k(mid) < k:
                l = mid+1
            else:
                r = mid
        return l

    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        heap = []
        for i in range(min(k, len(nums1))):
            heapq.heappush(heap, (nums1[i]+nums2[0], nums1[i], nums2[0], 0))
        
        res = []
        while k > 0 and heap:
            sum, n1, n2, idx = heapq.heappop(heap)
            res.append([n1, n2])
            if idx + 1 < len(nums2):
                heapq.heappush(heap, (n1+nums2[idx + 1], n1, nums2[idx + 1], idx + 1))
            k -= 1
        return res

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        total = len(nums1)+len(nums2)
        half = total // 2
        if len(nums2) < len(nums1):
            nums1, nums2 = nums2, nums1
        l = 0
        r = len(nums1)-1
        while True:
            i = (l+r) //2
            j = half-i-2
            Aleft = nums1[i] if i >= 0 else float("-inf")
            Aright = nums1[i+1] if i+1 < len(nums1) else float("inf")
            Bleft = nums2[j] if j >= 0 else float("-inf")
            Bright = nums2[j+1] if j+1 < len(nums2) else float("inf")

            if Aleft <= Bright and Bleft <= Aright:
                if total%2 == 0:
                    return (max(Aleft, Bleft)+min(Aright, Bright)) /2.0
                else:
                    return min(Aright, Bright) * 1.0
            elif Aleft >Bright:
                r = i-1
            else:
                l = i+1

    def subarraysWithKDistinct(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def atMost(nums, k):
            res = 0
            d = dict()
            left = 0
            right = 0
            for i in range(len(nums)):
                if nums[i] not in d:
                    d[nums[i]] = 1
                else:
                    d[nums[i]] += 1
                
                while len(d) > k:
                    d[nums[left]] -= 1
                    if d[nums[left]] == 0:
                        del d[nums[left]]
                    left += 1
                
                res += right-left+1
                right +=1
                
            return res

        return atMost(nums, k) - atMost(nums, k-1)

    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        heap = []
        res = []
        for num in nums:
            heapq.heappush(heap, num)
        while heap:
            res.append(heapq.heappop(heap))
        return res

    def getOrder(self, tasks):
        """
        :type tasks: List[List[int]]
        :rtype: List[int]
        """
        res = []
        heap = []
        tasks = sorted((task, i) for i, task in enumerate(tasks))
        i = 0
        curr = 0
        
        for (e, p), i in tasks:
            while heap and curr < e:
                proc, idx, en = heapq.heappop(heap)
                res.append(idx)
                curr = max(curr, en) + proc
            heapq.heappush(heap, (p, i, e))
        while heap:
            proc, idx, en = heapq.heappop(heap)
            res.append(idx)

        return res

    def minKBitFlips(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        flip_count = 0
        flip_queue = collections.deque()
        for i,num in enumerate(nums):
            condition1 = num==0 and len(flip_queue)%2==0
            condition2 = num==1 and len(flip_queue)%2!=0
            if condition1 or condition2:
                flip_count += 1
                flip_queue.append(i+k-1)
            if flip_queue and i >= flip_queue[0]:
                flip_queue.popleft()
        return -1 if flip_queue else flip_count

    def minWindow(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: str
        """
        if len(s2) > len(s1):
            return ""
        def find_end(start1):
            start2 = 0
            while start1 < len(s1):
                if s1[start1] == s2[start2]:
                    start2+=1
                    if start2 == len(s2):
                        break
                start1+=1
            return start1 if start2 == len(s2) else None
        def better_start(end1):
            end2 = len(s2)-1
            while end2 >= 0:
                if s1[end1] == s2[end2]:
                    end2 -=1
                end1 -=1
            return end1+1

        start1 = 0
        res = ""
        length = float("inf")
        while start1 < len(s1):
            end = find_end(start1)
            if end == None:
                break
            b_start = better_start(end)
            if end-b_start+1 < length:
                length = end-b_start+1
                res = s1[b_start:end+1]
            start1 = b_start+1
        return res

    def minimumDeviation(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        heap = []
        for num in nums:
            if num % 2 !=0:
                heap.append((num, num*2))
            else:
                temp = num
                while temp%2==0:
                    temp //=2
                heap.append((temp, max(temp*2, num))) 
        max_num = max([i for i, j in heap])
        heapq.heapify(heap)

        res = float("inf")
        while len(heap) == len(nums):
            MIN, MAX = heapq.heappop(heap)
            if max_num - MIN < res:
                res = max_num - MIN
            if MIN < MAX:
                heapq.heappush(heap, (MIN*2, MAX))
                max_num = max(max_num, MIN*2)
        return res

    def findMaximizedCapital(self, k, w, profits, capital):
        """
        :type k: int
        :type w: int
        :type profits: List[int]
        :type capital: List[int]
        :rtype: int
        """
        projects = [(capital[i], profits[i]) for i in range(len(profits))]
        projects.sort()
        heap = []
        for _ in range(k):
            while projects and projects[0][0] <= w:
                capital, profit = projects.pop(0)
                heapq.heappush(heap, -profit)
            if not heap:
                break
            w -= heapq.heappop(heap)
        return w
    
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = 1
        right = n
        while left < right:
            mid = (left+right)//2
            if isBadVersion(mid) == False:
                left=mid+1
            else:
                right=mid
        return left
    
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        if len(nums) == 1:
            return nums[0]
        while left<=right:
            mid = (left+right)//2
            if (mid+1 <len(nums) and nums[mid]!=nums[mid+1]) and (mid-1 >= 0 and nums[mid]!=nums[mid-1]):
                return nums[mid]
            a = mid%2==0 and nums[mid]==nums[mid-1]
            b = mid%2!=0 and nums[mid]==nums[mid+1]
            if a or b:
                right = mid-1
            else:
                left = mid+1
        return nums[right]

    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.accu = []
        self.total = 0
        for weight in w:
            self.total+=weight
            self.accu.append(self.total)

    def pickIndex(self):
        """
        :rtype: int
        """
        a = random.randint(1, self.total)
        return bisect.bisect_left(self.accu, a)
    
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left]<=nums[mid]: 
                if nums[left]<=target<nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                if nums[mid]<target<=nums[right]:
                    left = mid+1
                else:
                    right = mid-1

        return -1
    
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            print(left, mid, right)
            if nums[mid] == target or nums[left] == target or nums[right] == target:
                return True
            if nums[left]<=nums[mid]: 
                if nums[left]<=target<=nums[mid]:
                    right = mid-1
                else:
                    left += 1
            else:
                if nums[mid]<=target<=nums[right]:
                    left = mid+1
                else:
                    right -= 1

        return False
    
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        people.sort()
        low = 0
        high = len(people)-1
        res = 0
        while low<=high:
            if people[low]+people[high] <= limit:
                res += 1
                low += 1
                high -= 1
            else:
                res += 1
                high -= 1
        return res
    
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        trip = 0
        curr = 0
        start = 0
        for i in range(len(gas)):
            trip += gas[i] - cost[i]
            curr += gas[i] - cost[i]
            if curr < 0:
                curr = 0
                start = i+1
        if start<len(gas) and trip >= 0:
            return start
        return -1
    
    def minRefuelStops(self, target, startFuel, stations):
        """
        :type target: int
        :type startFuel: int
        :type stations: List[List[int]]
        :rtype: int
        """
        stations.append([target, 0])
        res = 0
        prev = 0
        currfuel = startFuel
        available = []
        for position, fuel in stations:
            distance = position - prev
            prev = position
            if currfuel < distance:
                while available and currfuel < distance:
                    gas = heapq.heappop(available)*-1
                    currfuel += gas
                    res += 1
                if currfuel < distance:
                    return -1

            currfuel -= distance
            heapq.heappush(available, -fuel)
        return res

    def twoCitySchedCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        SUM_B = sum(j for i, j in costs)
        diff = sorted(i-j for i, j in costs)
        save = sum(diff[:len(costs)//2])
        return SUM_B+save

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        if n == 2:
            return 2
        
        n1 = 1
        n2 = 2
        res = 0
        for i in range(3, n+1):
            res = n1+n2
            n1, n2 = n2, res
            
        return res

    def tribonacci(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        if n <= 2:
            return 1
        n0 = 0
        n1 = 1
        n2 = 1
        res = 0
        for i in range(3, n+1):
            res = n0+n1+n2
            n0, n1, n2 = n1, n2, res
        return res
    
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        total = sum(nums)
        if total % 2 != 0:
            return False
        half = total //2
        dp = [False] * (half+1)
        dp[0] = True
        for num in nums:
            for i in range(half, num-1, -1):
                dp[i] = dp[i] or dp[i-num]
        return dp[half]

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [float("inf")] * (amount+1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount+1):
                dp[i] = min(dp[i], dp[i-coin]+1)
        return -1 if dp[amount] == float("inf") else dp[amount]

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        last = len(nums)-1
        for i in range(len(nums)-2, -1, -1):
            if nums[i]+i >= last:
                last = i
        return last == 0
    
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0]*len(nums)
        j = 0
        for i in range(1, len(nums)):
            while j < len(nums) and j+nums[j] < i:
                j+=1
            dp[i] = dp[j]+1 
        return dp[-1]

    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start = 0
        nums.sort()
        for num in nums:
            if num == start:
                start += 1
            else:
                return start
        return start
    
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        l = 0
        r = len(arr)-1
        while l <= r:
            m = (l+r)//2
            if arr[m]-m-1 < k:
                l = m+1
            else:
                r = m-1
        return l+k

    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i]<=n and nums[nums[i]-1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n+1
    
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return 0
        nums.sort()
        gap = 0
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] > gap:
                gap = nums[i] - nums[i-1]
        return gap
    
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        def comparator(x1, x2):
            if x1+x2 > x2+x1:
                return 1
            elif x1+x2 < x2+x1:
                return -1
            return 0
        
        nums = [str(num) for num in nums]
        nums.sort(key=functools.cmp_to_key(comparator), reverse=True)
        return "0" if nums[0] == "0" else "".join(nums)
    
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = sorted(nums, reverse=True)
        for i in range(1, len(nums), 2):
            nums[i] = n.pop(0)
        for i in range(0, len(nums), 2):
            nums[i] = n.pop(0)

    def rearrangeBarcodes(self, barcodes):
        """
        :type barcodes: List[int]
        :rtype: List[int]
        """
        d = collections.Counter(barcodes)
        heap = []
        for k, v in d.items():
            heapq.heappush(heap, (-v, k))
        res = [None]*len(barcodes)
        i = 0
        while heap:
            count, key = heapq.heappop(heap)
            while count != 0:
                if i >= len(res):
                    i = 1
                res[i] = key
                i += 2
                count += 1

        return res

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for paren in s:
            if paren == "(" or paren == "[" or paren == "{":
                stack.append(paren)
            elif paren == ")":
                if stack and stack[-1] == "(":
                    stack.pop()
                else:
                    return False
            elif paren == "]": 
                if stack and stack[-1] == "[":
                    stack.pop()
                else:
                    return False
            elif paren == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                else:
                    return False
        
        return len(stack) == 0

    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        stack.append(s[0])
        for i in range(1, len(s)):
            if stack and stack[-1] == s[i]:
                stack.pop()
            else:
                stack.append(s[i])
        return ''.join(stack)

    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        ss = list(s)
        for i in range(len(ss)):
            if ss[i] == "(":
                stack.append(i)
            elif ss[i] == ")":
                if stack:
                    stack.pop()
                else:
                    ss[i] = ""
        if stack:
            for idx in stack:
                ss[idx] = ""


        return "".join(ss)

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        preCourse_canTake = collections.defaultdict(list)
        degree = [0] * numCourses
        for course, pre in prerequisites:
            preCourse_canTake[pre].append(course)
            degree[course] += 1
        
        queue = deque()
        for i in range(len(degree)):
            if degree[i] == 0:
                queue.append(i)
        while queue:
            curr = queue.popleft()
            for canTake in preCourse_canTake[curr]:
                degree[canTake] -= 1
                if degree[canTake] == 0:
                    queue.append(canTake)
    
        for i in range(len(degree)):
            if degree[i] != 0:
                return False
        return True
    
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        preCourse_canTake = collections.defaultdict(list)
        degree = [0] * numCourses
        for course, pre in prerequisites:
            preCourse_canTake[pre].append(course)
            degree[course] += 1
        
        queue = deque()
        for i in range(len(degree)):
            if degree[i] == 0:
                queue.append(i)
        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for canTake in preCourse_canTake[curr]:
                degree[canTake] -= 1
                if degree[canTake] == 0:
                    queue.append(canTake)
                
        return res if len(res) == numCourses else []

##################################
    def num_perfect_pairs(arr):
        arr.sort(key=lambda x:abs(x))
        left = 0
        right = 1
        cnt = 0
        while left < len(arr)-1 and right < len(arr):
            x = abs(arr[left])
            y = abs(arr[right])
            if y <= 2*x:
                cnt += right-left
                right += 1
            else:
                left += 1
        return cnt

    def job_execution(n, executionTime, x, y):
        executionTime.sort(reverse=True)
        i = 0
        cnt = 0
        while True:
            if executionTime[i] > 0:
                executionTime[i] -= x
                for j in range(1, len(executionTime)):
                    executionTime[j] -= y
                executionTime.sort(reverse=True)
                cnt += 1
            else:
                return cnt
            
    def string_formation(self, words, target):
        """
        :type words: List[str]
        :type target: str
        :rtype: int
        """
        dp=[0]*(len(target)+1)
        dp[0]=1
        count=[[0]*26 for _ in range(len(words[0]))]
        for i in range(len(words[0])):
            for word in words:
                count[i][ord(word[i])-ord('a')]+=1
        for i in range(len(words[0])):
            for j in range(len(target)-1,-1,-1):
                dp[j+1]+= dp[j]*count[i][ord(target[j])-ord('a')]
                dp[j+1]%= (10**9+7)

        return dp[-1] 

    def max_array_value(a):
        i, j = min(a), max(a)
        while i < j:
            m = (i + j) // 2
            avail = 0
            canMake = True
            for el in a:
                if el <= m:
                    avail += m - el
                else:
                    mustSubtract = max(0, el - m)
                    if mustSubtract > avail:
                        canMake = False
                        break
                    avail -= mustSubtract
            if canMake:
                j = m
            else:
                i = m + 1
        return i
    
    def task_scheduling(n, c, t):
        def f(i, j, memo):
            if i == n:
                return [float('inf'), 0][j >= 0]
            if (i, j) in memo:
                return memo[(i, j)]
        
            result = min(c[i] + f(i + 1, j + t[i], memo), f(i + 1, j - 1, memo))
            memo[(i, j)] = result
            return result
        memo = {}
        return f(0, 0, memo)
    
    def cross_the_threshold(n, initEnergy, threshold):
        res = 0
        s = sum(initEnergy)
        while s > threshold:
            res += 1
            temp = 0
            for i in range(n):
                initEnergy[i] -= 1
                if initEnergy[i] > 0:
                    temp += initEnergy[i]
            s = temp
        return res -1

    #https://www.geeksforgeeks.org/number-of-distinct-words-of-size-n-with-at-most-k-contiguous-vowels/
    def power(x, y, p):
        result = 1
        x = x % p
        if x == 0:
            return 0
        while y > 0:
            if y % 2 == 1:
                result = (result * x) % p
            y = y // 2
            x = (x * x) % p
        return result

    def power(x, y, p):
        res = 1
        x = x % p
        if (x == 0):
            return 0
        while (y > 0):
            if (y & 1):
                res = (res * x) % p
            y = y >> 1
            x = (x * x) % p
        return res
 
    def kvowelwords(N, K):
        i, j = 0, 0
        MOD = 1000000007
        dp = [[0 for i in range(K + 1)] for i in range(N + 1)]
        sum = 1
        for i in range(1, N + 1):
            dp[i][0] = sum * 21
            dp[i][0] %= MOD
            sum = dp[i][0]
            for j in range(1, K + 1):
                if (j > i):
                    dp[i][j] = 0
                elif (j == i):
                    dp[i][j] = power(5, i, MOD)
                    #dp[i][j] = (5 ** i) % MOD
                else:
                    dp[i][j] = dp[i - 1][j - 1] * 5
                dp[i][j] %= MOD
                sum += dp[i][j]
                sum %= MOD
    
        return sum

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(nums, index, path, res):
            res.append(path)
            for i in range(index, len(nums)):
                dfs(nums, i+1, path+[nums[i]], res)
        res = []
        dfs(sorted(nums), 0, [], res)
        return res
    
        def backtrack(start, end, tmp):
            res.append(tmp[:])
            for i in range(start, end):
                tmp.append(nums[i])
                backtrack(i+1, len(nums), tmp)
                tmp.pop()
                
        res = []
        backtrack(0, len(nums), [])
        return res

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(start, end, tmp):
            res.append(tmp[:])
            for i in range(start, end):
                if i != start and nums[i] == nums[i-1]:
                    continue
                tmp.append(nums[i])
                backtrack(i+1, len(nums), tmp)
                tmp.pop()

        nums.sort()     
        res = []
        backtrack(0, len(nums), [])
        return res

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res = []
        def backtrack(temp, digits):
            if not digits:
                return res.append(temp)
            for letter in phone[digits[0]]:
                backtrack(temp+letter, digits[1:])
        backtrack("", digits)
        return res
    
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def backtrack(open, close, temp):
            if open == close == n:
                return res.append("".join(temp))
            if open < n:
                temp.append("(")
                backtrack(open+1, close, temp)
                temp.pop()
            if close < n and open>close:
                temp.append(")")
                backtrack(open, close+1, temp)
                temp.pop()

        res = []
        backtrack(0, 0, [])
        return res

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(nums, temp, res):
            if not nums:
                res.append(temp[:])
            
            for i in range(len(nums)):
                newNums = nums[:i] + nums[i+1:]
                temp.append(nums[i])
                backtrack(newNums, temp, res)
                temp.pop()

        res = []
        backtrack(nums, [], res)
        return res
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(nums, temp, res):
            if not nums:
                res.append(temp[:])
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i-1]:
                    continue
                newNums = nums[:i]+nums[i+1:]
                temp.append(nums[i])
                backtrack(newNums, temp, res)
                temp.pop()
        nums.sort()
        res = []
        backtrack(nums, [], res)
        return res
    
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        def backtrack(remain, temp, idx):
            if remain == 0:
                res.append(temp[:])
            for i in range(idx, n+1):
                temp.append(i)
                backtrack(remain-1, temp, i+1)
                temp.pop()
        res = []
        backtrack(k, [], 1)
        return res
    
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def backtrack(idx, temp, target):
            if target < 0:
                return 
            if target == 0:
                return res.append(temp[:])
            for i in range(idx, len(candidates)):
                temp.append(candidates[i])
                backtrack(i, temp, target-candidates[i])
                temp.pop()
        res = []
        backtrack(0, [], target)
        return res

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def backtrack(idx, temp, target):
            if target < 0:
                return 
            if target == 0:
                return res.append(temp[:])
            for i in range(idx, len(candidates)):
                if i!=idx and candidates[i] == candidates[i-1]:
                    continue
                temp.append(candidates[i])
                backtrack(i+1, temp, target-candidates[i])
                temp.pop()
        res = []
        candidates.sort()
        backtrack(0, [], target)
        return res
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        def backtrack(nums, k, n, temp, res):
            if k < 0 or n < 0:
                return 
            if k == 0 and n == 0:
                res.append(temp[:])
            for i in range(len(nums)):
                newNums = nums[i+1:]
                backtrack(newNums, k-1, n-nums[i], temp+[nums[i]], res)
        res = []
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        backtrack(nums, k, n, [], res)
        return res
    
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def backtrack(s, temp, res):
            if not s:
                return res.append(temp[:])
            for i in range(1, len(s)+1):
                if s[:i] == s[i-1::-1]:
                    temp.append(s[:i])
                    backtrack(s[i:], temp, res)
                    temp.pop()
        res = []
        backtrack(s, [], res)
        return res
    
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # if not root:
        #     return 0
        # return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
    
        if not root:
            return 0
        level = 0
        level_nodes = 1
        q = deque()
        q.append(root)
        while q:
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            level_nodes -= 1
            if level_nodes == 0:
                level += 1
                level_nodes = len(q)
        return level
    
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.right)
        self.invertTree(root.left)
        
        return root

    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.ans = max(self.ans, left+right)
            return max(left, right) + 1
        dfs(root)
        return self.ans

    def __init__(self):
        self.prev = None
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        if not root:
            return None
        self.flatten(root.right)
        self.flatten(root.left)
        root.left = None
        root.right = self.prev
        self.prev = root

    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.res = float("-inf")
        def dfs(node):
            if not node:
                return 0
            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)
            self.res = max(self.res, left+right+node.val)
            return node.val + max(left, right)
        dfs(root)
        return self.res

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        s = []
        def dfs(root):
            if not root:
                return s.append('n')
            s.append(str(root.val))
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return ' '.join(s)      

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        q = deque(data.split())
        def dfs():
            node = q.popleft()
            if node == 'n':
                return None
            root = TreeNode(node)
            root.left = dfs()
            root.right = dfs()
            return root
        return dfs()
    
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def bfs(k):
            if len(k) == 0:
                return
            temp = []
            new = []
            for node in k:
                temp.append(node.val)
                if node.left:
                    new.append(node.left)
                if node.right:
                    new.append(node.right)
            res.append(temp)
            bfs(new)
        res = []
        if root:
            bfs([root])
        return res
    
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def bfs(k, flag):
            if len(k) == 0:
                return
            temp = []
            new = []
            for node in k:
                temp.append(node.val)
                if node.left:
                    new.append(node.left)
                if node.right:
                    new.append(node.right)
            if flag == 1:    
                res.append(temp)
            else:
                res.append(temp[::-1])
            flag *= -1
            bfs(new, flag)
        res = []
        flag = 1
        if root:
            bfs([root], flag)
        return res

    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        q = [root]
        while q:
            curr = q.pop(0)
            if curr.left and curr.right:
                curr.left.next = curr.right
                if curr.next:
                    curr.right.next = curr.next.left
                q.append(curr.left)
                q.append(curr.right)
        return root

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        stack = [root]
        parent = {root: None}
        while not(p in parent and q in parent):
            node = stack.pop()
            if node.left:
                stack.append(node.left)
                parent[node.left] = node
            if node.right:
                stack.append(node.right)
                parent[node.right] = node
        common = set()
        while p:
            common.add(p)
            p = parent[p]
        while q not in common:
            q = parent[q]
        return q
    
    def verticalTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        dic = collections.defaultdict(list)
        
        def helper(placement, level, node):
            if not node:
                return
            dic[placement].append((level, node.val))
            helper(placement-1, level+1, node.left)
            helper(placement+1, level+1, node.right)
        helper(0, 0, root)
        res = []
        for i in sorted(dic.keys()):
            temp = []
            for j in sorted(dic[i]):
                temp.append(j[1])
            res.append(temp)
        return res

    def __init__(self):
        self.data = [None]*1000001

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        self.data[key] = value

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        v = self.data[key]
        return v if v != None else -1

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        self.data[key] = None
    
    def __init__(self):
        self.dic = {}
    def shouldPrintMessage(self, timestamp, message):
        """
        :type timestamp: int
        :type message: str
        :rtype: bool
        """
        if message not in self.dic:
            self.dic[message] = timestamp+9
            return True
        else:
            val = self.dic[message]
            if timestamp > val:
                self.dic[message] = timestamp+9
                return True
            else:
                return False
    
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        stack = []
        d = {}
        for num in nums2:
            while stack and num > stack[-1]:
                d[stack.pop()] = num
            stack.append(num)
        while stack:
            d[stack.pop()] = -1
        for num in nums1:
            res.append(d[num])
        return res
    
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s1 = []
        s2 = []
        for n in s:
            s1.append(s.index(n))
        for n in t:
            s2.append(t.index(n))
        print(s1, s2)
        if s1 == s2:
            return True
        return False

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        odd = set()
        for letter in s:
            if letter not in odd:
                odd.add(letter)
            else:
                odd.remove(letter)
        if len(odd) == 0:
            return len(s)
        else:
            return len(s) - len(odd) + 1
        
    def fractionToDecimal(self, numerator, denominator):

        n, remainder = divmod(abs(numerator), abs(denominator))
        sign = '-' if numerator*denominator < 0 else ''
        result = [sign+str(n), '.']
        remainders = {}

        while remainder > 0 and remainder not in remainders:
            remainders[remainder] = len(result)
            n, remainder = divmod(remainder*10, abs(denominator))
            result.append(str(n))

        if remainder in remainders:
            idx = remainders[remainder]
            result.insert(idx, '(')
            result.append(')')

        return ''.join(result).rstrip(".")

    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        d = collections.defaultdict(int)
        for letter in magazine:
            if letter not in magazine:
                d[letter] = 1
            else:
                d[letter] += 1
        for word in ransomNote:
            if d[word] <= 0:
              return False
            d[word] -= 1
        return True

    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        odd = True
        d = {}
        for letter in s:
            if letter not in d:
                d[letter] = 1
            else:
                d[letter] += 1
        
        for k, v in d.items():
            if v % 2 == 0:
                continue
            elif v % 2 != 0 and odd == True:
                odd = False
            else:
                return False
        return True

    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        for letter in s:
            if letter in d:
                d[letter] += 1
            else:
                d[letter] = 1
        for idx in range(len(s)):
            if d[s[idx]] == 1:
                return idx
        return -1

    def groupAnagrams(self, words):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        d = collections.defaultdict(list)
        for word in words:
            key = [0]*26
            for letter in word:
                key[ord(letter)-ord('a')] += 1
            d[tuple(key)].append(word)
        res = []
        for v in d.values():
            res.append(v)
        return res

    def __init__(self, n):
        """
        :type n: int
        """
        self.n = n
        self.row = [0] * n
        self.col = [0] * n
        self.dia = 0
        self.anti = 0

    def move(self, row, col, player):
        """
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        val = 1 if player == 1 else -1

        self.row[row] += val
        self.col[col] += val
        if row == col:
            self.dia += val
        if row == self.n-1-col:
            self.anti += val
        if self.row[row] == self.n or self.col[col] == self.n or self.dia == self.n or self.anti == self.n:
            return 1
        if abs(self.row[row]) == self.n or abs(self.col[col]) == self.n or abs(self.dia) == self.n or abs(self.anti) == self.n:
            return 2

        return 0
    
    def __init__(self):
        self.count = collections.defaultdict(int)
        self.idx = 0
        self.heap = []  
    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.count[val] += 1
        self.idx += 1
        heappush(self.heap, (-self.count[val], -self.idx, val))
    def pop(self):
        """
        :rtype: int
        """
        _, _, val = heappop(self.heap)
        self.count[val] -=1
        return val
    
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        p_counter = Counter(p)
        s_counter = dict()
        res = []
        for i in range(len(s)-len(p)+1):
            if i == 0:
                s_counter = Counter(s[:len(p)])
            else:
                s_counter[s[i-1]] -= 1
                s_counter[s[i+len(p)-1]] += 1
            if len(p_counter-s_counter) == 0:
                res.append(i)
        return res

    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        l1 = list1
        l2 = list2
        dummy = ListNode(0)
        curr = dummy
        while l1!=None and l2!=None:
            if l1 and l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
                curr = curr.next
            else:
                curr.next = l2
                l2 = l2.next
                curr = curr.next
        if l1:
            curr.next = l1
        else:
            curr.next = l2
        return dummy.next

    class Node(object):
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None
class BrowserHistory(object):
    def __init__(self, homepage):
        """
        :type homepage: str
        """      
        self.root = Node(homepage)
    def visit(self, url):
        """
        :type url: str
        :rtype: None
        """
        new = Node(url)
        self.root.next = new
        new.prev = self.root
        self.root = new
    def back(self, steps):
        """
        :type steps: int
        :rtype: str
        """
        while steps and self.root.prev:
            steps -= 1
            self.root = self.root.prev
        return self.root.val
    def forward(self, steps):
        """
        :type steps: int
        :rtype: str
        """
        while steps and self.root.next:
            steps -= 1
            self.root = self.root.next
        return self.root.val
    
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        curNum = 0
        curString = ''
        for c in s:
            if c == "[":
                stack.append(curString)
                stack.append(curNum)
                curNum = 0
                curString = ''
            elif c == "]":
                time = stack.pop()
                prev = stack.pop()
                curString = prev + time*curString
            elif c.isdigit():
                curNum = curNum*10+int(c)
            else:
                curString += c
        return curString
    
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        stack = []
        for c in path.split("/"):
            if c == "..":
                if stack:
                    stack.pop()
            elif c == "." or not c:
                continue
            else:
                stack.append(c)     
        return '/'+'/'.join(stack)
    
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        stack = []
        idx = 0
        for num in pushed:
            stack.append(num)
            while stack and stack[-1] == popped[idx]:
                stack.pop()
                idx += 1
        return True if len(stack) == 0 else False

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        stack = [-1]
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    res = max(res, i-stack[-1])
        return res

class NestedIterator(object):
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = nestedList[::]
    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop(0).getInteger()
    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
            first = self.stack[0]
            if first.isInteger():
                return True
            self.stack = first.getList()+self.stack[1:]
        return False
    
    def removeDuplicates(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        stack = []
        stack.append((s[0], 1))
        for i in range(1, len(s)):
            if stack and stack[-1][0] == s[i]:
                letter, num = stack.pop()
                num += 1
                if num == k:
                    continue
                else:
                    stack.append((letter, num))
            else:
                stack.append((s[i], 1))
        if stack:
            temp = []
            for l, v in stack:
                for i in range(v):
                    temp.append(l)
            return "".join(temp)
        return ""

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        stack = []
        num = 0
        res = 0
        sign = 1
        for c in s:
            if c.isdigit():
                num = num*10+int(c)
            elif c in "-+":
                res += sign*num
                sign = -1 if c =="-" else 1
                num = 0
            elif c == "(":
                stack.append(res)  
                stack.append(sign)
                sign = 1
                res = 0
            elif c == ")":
                res += sign*num
                res *= stack.pop()
                res += stack.pop()
                num = 0
        return res + sign*num

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        num = 0
        res = 0
        pre_op = '+'
        s+='+'
        stack = []
        for c in s:
            if c.isdigit():
                num = num*10 + int(c)
            elif c == ' ':
                    pass
            else:
                if pre_op == '+':
                    stack.append(num)
                elif pre_op == '-':
                    stack.append(-num)
                elif pre_op == '*':
                    operant = stack.pop()
                    stack.append((operant*num))
                elif pre_op == '/':
                    operant = stack.pop()
                    stack.append(math.trunc(operant/num))
                num = 0
                pre_op = c
        return sum(stack)

    def uniqueOccurrences(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        d = dict()
        for ele in arr:
            if ele not in d:
                d[ele] = 1
            else:
                d[ele] += 1
        
        a = [0]*1001
        for k, v in d.items():
            if a[v] == 0:
                a[v] = v
            else:
                return False
        return True

    def subarraysDivByK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        pre_sum = 0
        d = {0:1}
        res = 0
        for num in nums:
            pre_sum += num
            key = pre_sum % k
            if key in d:
                res += d[key]
                d[key] += 1
                continue
            d[key] = 1
        return res
    
    def minimumRounds(self, tasks: List[int]) -> int:
        d = dict()
        for t in tasks:
            if t not in d:
                d[t] = 1
            else:
                d[t] += 1
        res = 0
        for k, v in d.items():
            if v == 1:
                return -1
            #res += math.ceil(v/3)
            #res += (v+2)//3
            while v % 3 != 0:
                res += 1
                v -= 2
            res += (v//3)
        return res

    class DLNode(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None  
class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.d = dict()
        self.head = DLNode(0, 0)
        self.tail = DLNode(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    def move_to_head(self, node):
        n = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = n
        n.prev = node
    def delete_from_list(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
    def remove_tail(self):
        if len(self.d) == 0: 
            return
        node = self.tail.prev
        self.delete_from_list(node)
        del self.d[node.key]
    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.d:
            return -1
        node = self.d[key]
        self.delete_from_list(node)
        self.move_to_head(node)
        return node.val
        
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key not in self.d:
            if len(self.d) >= self.capacity:
                self.remove_tail()
            new = DLNode(key, value)
            self.d[key] = new 
            self.move_to_head(new)
        else:
            node = self.d[key]
            self.delete_from_list(node)
            self.move_to_head(node)
            node.val = value
    
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        d = dict()
        for num in nums:
            if num not in d:
                d[num] = 1
            else:
                return True
        return False
    
class SnapshotArray(object):

    def __init__(self, length):
        """
        :type length: int
        """
        self.arr = [ [[-1, 0]] for i in range(length)]
        self.id = 0
    def set(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        self.arr[index].append([self.id, val])
    def snap(self):
        """
        :rtype: int
        """
        self.id += 1
        return self.id - 1
    def get(self, index, snap_id):
        """
        :type index: int
        :type snap_id: int
        :rtype: int
        """
        i = bisect.bisect_right(self.arr[index], [snap_id, float("inf")]) -1
        return self.arr[index][i][1]

class RandomizedSet(object):
    def __init__(self):
        self.arr = []
        self.map = {}
    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if val not in self.map:
            self.map[val] = len(self.arr)
            self.arr.append(val)
            return True
        return False
    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if val not in self.map:
            return False
        idx = self.map[val]
        arr_last = self.arr[-1]
        arr_last_idx = len(self.arr)-1
        self.arr[idx] = arr_last
        self.map[arr_last] = idx
        self.arr.pop()
        del self.map[val]
        return True 
    def getRandom(self):
        """
        :rtype: int
        """
        return random.choice(self.arr)
    
class MinStack(object):
    def __init__(self):   
        self.stack = []
        self.min = None
    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        if self.min == None:
            self.min = val 
        if val < self.min:
            self.min = val
        self.stack.append([val, self.min])
    def pop(self):
        """
        :rtype: None
        """   
        if self.stack:
            self.stack.pop()
            if not self.stack:
                self.min = None
            else:
                self.min = self.stack[-1][1]
    def top(sself):
        """
        :rtype: int
        """    
        if self.stack:
            return self.stack[-1][0]
    def getMin(self):
        """
        :rtype: int
        """
        if self.stack:
            return self.stack[-1][1]
        
class TimeMap(object):
    def __init__(self):
        self.map = collections.defaultdict(list)
    def set(self, key, value, timestamp):
        """
        :type key: str
        :type value: str
        :type timestamp: int
        :rtype: None
        """    
        self.map[key].append([timestamp, value])
    def get(self, key, timestamp):
        """
        :type key: str
        :type timestamp: int
        :rtype: str
        """
        val = self.map[key]
        if not val:
            return ""
        left = 0
        right = len(val)-1
        while left < right:
            mid = (left+right+1)/2
            if val[mid][0] == timestamp:
                return val[mid][1]
            elif val[mid][0] < timestamp:
                left = mid 
            else:
                right = mid - 1
        return val[left][1] if val[left][0] <= timestamp else ""
    
class OrderedStream(object):

    def __init__(self, n):
        """
        :type n: int
        """
        self.pointer = 0
        self.arr = [None] * n
    def insert(self, idKey, value):
        """
        :type idKey: int
        :type value: str
        :rtype: List[str]
        """
        idKey -= 1
        self.arr[idKey] = value
        if self.pointer < idKey:
            return []
        else:
            while self.pointer < len(self.arr) and self.arr[self.pointer] != None:
                self.pointer += 1
            return self.arr[idKey:self.pointer]

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l = 0
        r = len(nums) - 1
        while l <= r:
            m = (l+r) /2 
            if nums[m] == target:
                return m
            elif nums[m] < target:
                l = m + 1
            else:
                r = m - 1
        return -1
    
    def shipWithinDays(self, weights, days):
        """
        :type weights: List[int]
        :type days: int
        :rtype: int
        """
        left = max(weights)
        right = sum(weights)
        while left < right:
            m = (left+right)/2
            need = 1
            curr = 0
            for w in weights:
                if curr+w > m:
                    need += 1
                    curr = 0
                curr += w
            if need > days:
                left = m + 1
            else:
                right = m
        return left
    
    def numSubseq(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        res = 0
        m = 10**9 + 7
        l = 0
        r = len(nums) - 1
        while l <= r:
            if nums[l] + nums[r] > target:
                r -= 1
            else:
                res += pow(2, r-l, m)
                l += 1
        return res % m

    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        n1 = set(nums1)
        res = set()
        for n in nums2:
            if n in n1:
                res.add(n)
        return list(res)
    
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        return bisect.bisect_left(nums, target)

    def minimumTime(self, time, totalTrips):
        """
        :type time: List[int]
        :type totalTrips: int
        :rtype: int
        """
        l = 1
        r = min(time) * totalTrips

        while l < r:
            m = (l+r)//2
            if sum(m//t for t in time) >= totalTrips:
                r = m
            else:
                l = m+1
        return l

    def minimizeArrayValue(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = max(nums)
        while left < right:
            m = (left+right) // 2
            print(m, left, right)
            if self.check(nums, m):
                right = m
            else:
                left = m + 1     
        return left
    def check(self, nums, m):
        have = 0
        for n in nums:
            if n <= m:
                have += m-n
            else:
                if have < n - m:
                    return False
                else:
                    have -= n-m
        return True

    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        d = dict()
        for idx, letter in enumerate(order):
            d[letter] = idx
        
        for i in range(1, len(words)):
            prev = words[i-1]
            curr = words[i]
            flag = 0
            for j in range(min(len(prev), len(curr))):
                if d[prev[j]] < d[curr[j]]:
                    flag = 1
                    break
                elif d[prev[j]] > d[curr[j]]:
                    return False
            if not flag and len(prev) > len(curr):
                return False
        return True

    def findAllRecipes(self, recipes, ingredients, supplies):
        """
        :type recipes: List[str]
        :type ingredients: List[List[str]]
        :type supplies: List[str]
        :rtype: List[str]
        """
        q = deque()
        for supply in supplies:
            q.append(supply)
        ingred_reci = collections.defaultdict(list)
        reci_need = collections.defaultdict(int)
        for i in range(len(ingredients)):
            for j in range(len(ingredients[i])):
                ingred_reci[ingredients[i][j]].append(recipes[i])
                reci_need[recipes[i]] += 1
        res = []
        while q:
            have = q.popleft()
            for reci in ingred_reci[have]:
                reci_need[reci] -= 1
                if reci_need[reci] == 0:
                    res.append(reci)
                    q.append(reci)
        return res

    def pancakeSort(self, arr):
        """
        :type arr: List[int]
        :rtype: List[int]
        """
        res = []
        for x in range(len(arr), 0, -1):
            idx = arr.index(x)
            res.append(idx+1)
            arr = arr[0:idx+1][::-1]+arr[idx+1:]
            arr = arr[0:x][::-1]+arr[x:]
            res.append(x)
            #res.extend([idx + 1, x])
        return res

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid:
            return 0
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == "1":
                    res += 1
                    self.bfs(grid, i, j)
        return res

    def bfs(self, grid, i, j):
        if i > len(grid)-1 or i < 0 or j < 0 or j > len(grid[0])-1 or grid[i][j] != "1":
            return 
        grid[i][j] = "0"
        self.bfs(grid, i+1, j)
        self.bfs(grid, i-1, j)
        self.bfs(grid, i, j-1)
        self.bfs(grid, i, j+1)

    def maxIceCream(self, costs, coins):
        """
        :type costs: List[int]
        :type coins: int
        :rtype: int
        """
        costs.sort()
        res = 0
        for c in costs:
            if coins-c >= 0:
                res += 1
                coins -= c
            else:
                break
        return res

    def minimumSemesters(self, n, relations):
        """
        :type n: int
        :type relations: List[List[int]]
        :rtype: int
        """
        course = [0] * (n+1)
        prev_next = collections.defaultdict(list)
        for pre, nex in relations:
            prev_next[pre].append(nex)
            course[nex] += 1
        
        q = deque()
        for i in range(1, n+1):
            if course[i] == 0:
                q.append(i)
        if not q:
            return -1
        res = 0
        while q:
            res += 1
            for i in range(len(q)):
                curr = q.popleft()
                courses = prev_next[curr]
                for i in range(len(courses)):
                    course[courses[i]] -= 1
                    if course[courses[i]] == 0:
                        q.append(courses[i])
        return res if sum(course) == 0 else -1
    
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def ok(seg):
            if len(seg) > 3 or len(seg) == 0 or (len(seg) > 1 and seg[0] == "0") or int(seg) > 255:
                return False
            return True
        res = []
        for i in range(1, 4):
            for j in range(i+1, i+4):
                for k in range(j+1, j+4):
                    seg1, seg2, seg3, seg4 = s[:i], s[i:j], s[j:k], s[k:]
                    if ok(seg1) and ok(seg2) and ok(seg3) and ok(seg4):
                        res.append(seg1 + "." + seg2 + "." + seg3 + "." + seg4)
        return res

     def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        visit = set()
        def dfs(i, j, idx):
            if idx == len(word):
                return True
            if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) or board[i][j] != word[idx] or (i, j) in visit:
                return False
            visit.add((i, j))
            res = dfs(i+1, j, idx+1) or dfs(i-1, j, idx+1) or dfs(i, j+1, idx+1) or dfs(i, j-1, idx+1)
            visit.remove((i, j))
            return res
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False

    def numTilePossibilities(self, tiles):
        """
        :type tiles: str
        :rtype: int
        """
        res = set()
        def dfs(temp, tiles):
            if temp:
                res.add(temp)
            for i in range(len(tiles)):
                dfs(temp+tiles[i], tiles[:i]+tiles[i+1:])
        dfs("", tiles)
        return len(res)
    
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        nums = range(1, n+1)
        res = []
        k -= 1
        while n > 0:
            n -= 1
            idx, k = divmod(k, math.factorial(n))
            res.append(str(nums[idx]))
            nums.remove(nums[idx])
        return "".join(res)

    def numSquarefulPerms(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def dfs(temp,num):
            if len(num)==0:
                self.count+=1
                return 
            for i in xrange(len(num)):
                if (i>0 and num[i]==num[i-1]) or (len(temp) > 0 and math.sqrt(num[i] + temp[-1]) % 1 != 0):
                    continue
                dfs(temp+[num[i]],num[:i]+num[i+1:])
        self.count = 0
        nums.sort()
        dfs([],nums)
        return self.count

    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        def backtrack(start, temp):
            if len(temp) >= 2:
                res.append(temp[:])
            used = set()
            for i in range(start, len(nums)):
                if temp and nums[i] < temp[-1]:  
                    continue 
                if nums[i] in used:
                    continue
                used.add(nums[i]) 
                temp.append(nums[i])
                backtrack(i+1, temp)
                temp.pop()
        backtrack(0, [])
        return res

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """  
        rows = defaultdict(set)
        cols = defaultdict(set)
        boxes = defaultdict(set)
        dots = deque()
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    dots.append((i, j))
                else:
                    rows[i].add(board[i][j])
                    cols[j].add(board[i][j])
                    boxes[(i//3, j//3)].add(board[i][j])
        def dfs():
            if not dots:
                return True
            row, col = dots[0]
            box = (row//3, col//3)
            for n in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                if n not in rows[row] and n not in cols[col] and n not in boxes[box]:
                    board[row][col] = n
                    rows[row].add(n)
                    cols[col].add(n)
                    boxes[box].add(n)
                    dots.popleft()
                    if dfs():
                        return True
                    else:
                        board[row][col] = "."
                        rows[row].discard(n)
                        cols[col].discard(n)
                        boxes[box].discard(n)
                        dots.appendleft((row, col))
            return False
        dfs()

    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        cols = set()
        dia = set()
        anti_dia = set()
        def backtrack(r, temp):
            if r == n:
                return res.append([ "."*i + "Q" + "."*(n-i-1) for i in temp])
            for c in range(n):
                if c not in cols and r-c not in dia and r+c not in anti_dia:
                    cols.add(c)
                    dia.add(r-c)
                    anti_dia.add(r+c)
                    temp.append(c)
                    backtrack(r+1, temp)
                    cols.remove(c)
                    dia.remove(r-c)
                    anti_dia.remove(r+c)
                    temp.pop()
                    
        backtrack(0, [])
        return res

    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        self.res = 0
        cols = set()
        dia = set()
        anti_dia = set()
        def backtrack(r, temp):
            if r == n:
                self.res += 1
                return
            for c in range(n):
                if c not in cols and r-c not in dia and r+c not in anti_dia:
                    cols.add(c)
                    dia.add(r-c)
                    anti_dia.add(r+c)
                    temp.append(c)
                    backtrack(r+1, temp)
                    cols.remove(c)
                    dia.remove(r-c)
                    anti_dia.remove(r+c)
                    temp.pop()
                    
        backtrack(0, [])
        return self.res
    
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        # num = bin(n)[2:]
        # if "00" in num or "11" in num :
        #     return False
        # return True
        num = bin(n)[2:]
        prev = num[0]
        for i in range(1, len(num)):
            if prev == "0" and num[i] == "0":
                return False
            elif prev == "1" and num[i] == "1":
                return False
            prev = num[i]
        return True

    def reverseBits(self, n):
        res = 0
        for i in range(32):
            res = (res << 1) + (n & 1)
            n >>= 1
        return res

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for num in nums:
            res ^= num
        return res

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ones = 0
        twos = 0
        for num in nums:
            ones = (ones ^ num) & ~twos
            twos = (twos ^ num) & ~ones
        return ones

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = 0
        for num in nums:
            res ^= num
        right = 1
        while (res & 1) == 0:
            res >>= 1
            right <<= 1
        one = 0
        two = 0
        for num in nums:
            if (right & num) != 0:
                one ^= num
            else:
                two ^= num
        return [one, two]

    def rangeBitwiseAnd(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        i = 0
        while left != right:
            left >>= 1
            right >>= 1
            i += 1
        return left << i

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        res = ""
        i = len(a)-1
        j = len(b)-1
        carry = 0
        while i >= 0 or j >= 0:
            sum = carry
            if i >= 0:
                sum += (ord(a[i]) - ord("0"))
            if j >= 0:
                sum += (ord(b[j]) - ord("0"))
            i -= 1
            j -= 1
            if sum > 1:
                carry = 1
            else:
                carry = 0
            res += str(sum%2)
        if carry:
            res += str(carry)
        return res[::-1]

    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        s = set(nums)
        for num in nums:
            if num - 1 in s:
                continue
            curr = 1
            while num + curr in s:
                curr += 1
            res = max(res, curr)
        return res

    
class Node(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
        self.freq = 1
class DDL(object):
    def __init__(self):
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
    def insert_to_head(self, node):
        n = self.head.next
        self.head.next = node
        node.prev = self.head
        n.prev = node
        node.next = n
        self.size += 1
    def remove_node(self, node):
        n = node.next
        p = node.prev
        p.next = n
        n.prev = p
        self.size -= 1
    def pop_tail(self):
        node = self.tail.prev
        self.remove_node(node)
        return node
class LFUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = {}
        self.freqTable = collections.defaultdict(DDL)
        self.capacity = capacity
        self.minFreq = 0

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.cache:
            return -1
        return self.update(self.cache[key], self.cache[key].val)
       
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.cache:
            self.update(self.cache[key], value)
        else:
            if len(self.cache) == self.capacity:
                tail = self.freqTable[self.minFreq].pop_tail()
                del self.cache[tail.key]
            new = Node(key, value)
            self.cache[key] = new
            self.minFreq = 1
            self.freqTable[1].insert_to_head(new)
    
    def update(self, node, value):
        node.val = value
        prevFreq = node.freq
        node.freq += 1
        self.freqTable[prevFreq].remove_node(node)
        self.freqTable[node.freq].insert_to_head(node)
        if prevFreq == self.minFreq and self.freqTable[prevFreq].size == 0:
            self.minFreq += 1
        return node.val

    def minDiffInBST(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """    
        stack = []
        node = root
        res = float("inf")
        prev = float("-inf")
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            res = min(res, node.val - prev)
            prev = node.val
            node = node.right
        return res
    
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        res = float("inf")
        prev = float("-inf")
        stack = []
        node = root
        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            res = min(res, abs(node.val-prev))
            prev = node.val
            node = node.right
        return res

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.isMirror(root.left, root.right)
    def isMirror(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return left.val == right.val and self.isMirror(left.left, right.right) and self.isMirror(left.right, right.left)

    def floodFill(self, image, sr, sc, color):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type color: int
        :rtype: List[List[int]]
        """
        q = deque()
        seen = set()
        seen.add((sr, sc))
        q.append((sr, sc))
        init = image[sr][sc]
        res = []
        while q:
            r, c = q.popleft()
            if image[r][c] == init:
                image[r][c] = color
            if r-1 >= 0 and image[r-1][c] == init and (r-1, c) not in seen:
                seen.add((r-1, c))
                q.append((r-1, c))
            if r+1 <= len(image)-1 and image[r+1][c] == init and (r+1, c) not in seen:
                seen.add((r+1, c))
                q.append((r+1, c))
            if c-1>=0 and image[r][c-1] == init and (r, c-1) not in seen:
                seen.add((r, c-1))
                q.append((r, c-1))
            if c+1 <= len(image[0])-1 and image[r][c+1] == init and (r, c+1) not in seen:
                seen.add((r, c+1))
                q.append((r, c+1))

        return image
    
class Trie(object):
    def __init__(self):
        self.trie = {}
    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        t = self.trie
        for w in word:
            if w not in t:
                t[w] = {}
            t = t[w]
        t["-"] = True
    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        t = self.trie
        for w in word:
            if w not in t:
                return False
            t = t[w]
        return "-" in t
    def startsWith(self, prefix):
        """
        :type prefix: str
        :rtype: bool
        """
        t = self.trie
        for w in prefix:
            if w not in t:
                return False
            t = t[w]
        return True

class Node(object):
    def __init__(self):
        self.children = {}
        self.isWord = False
class Trie(object):
    def __init__(self):
        self.trie = Node()
    def insert(self, word):
        """
        :type word: str
        :rtype: None
        """
        curr = self.trie
        for w in word:
            if w not in curr.children:
                curr.children[w] = Node()
            curr = curr.children[w]
        curr.isWord = True
    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        curr = self.trie
        for w in word:
            if w not in curr.children:
                return False
            curr = curr.children[w]
        return curr.isWord
    def startsWith(self, prefix):
        """
        :type prefix: str
        :rtype: bool
        """
        curr = self.trie
        for w in prefix:
            if w not in curr.children:
                return False
            curr = curr.children[w]
        return True

class Node(object):
    def __init__(self):
        self.children = {}
        self.isWord = False
class WordDictionary(object):
    def __init__(self):
        self.trie = Node()
    def addWord(self, word):
        """
        :type word: str
        :rtype: None
        """
        curr = self.trie
        for w in word:
            if w not in curr.children:
                curr.children[w] = Node()
            curr = curr.children[w]
        curr.isWord = True             
    def search(self, word):
        """
        :type word: str
        :rtype: bool
        """
        def dfs(t, index):
            if index == len(word):
                return t.isWord
            if word[index] == ".":
                for child in t.children.values():
                    if dfs(child, index+1):
                        return True
            if word[index] in t.children:
                return dfs(t.children[word[index]], index+1)
            return False
        return dfs(self.trie, 0)

class Node(object):
    def __init__(self):
        self.children = collections.defaultdict(Node)
        self.suggestion = []
    def add_suggestion(self, product):
        if len(self.suggestion) < 3:
            self.suggestion.append(product)
class Solution(object):
    def suggestedProducts(self, products, searchWord):
        """
        :type products: List[str]
        :type searchWord: str
        :rtype: List[List[str]]
        """
        products.sort()
        root = Node()
        for product in products:
            node = root
            for char in product:
                node = node.children[char]
                node.add_suggestion(product)
        res = []
        node = root
        for char in searchWord:
            node = node.children[char]
            res.append(node.suggestion)
        return res

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        visit = set()
        def dfs(i, j, idx):
            if idx == len(word):
                return True
            if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) or board[i][j] != word[idx] or (i, j) in visit:
                return False
            visit.add((i, j))
            res = dfs(i+1, j, idx+1) or dfs(i-1, j, idx+1) or dfs(i, j+1, idx+1) or dfs(i, j-1, idx+1)
            visit.remove((i, j))
            return 
        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, 0):
                    return True
        return False

class TrieNode():
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = False

class Trie():
    def __init__(self):
        self.root = TrieNode()
        self.num_of_words = 0
    
    def insert(self, word):
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True
        self.num_of_words += 1

class Solution(object):
    def dfs(self, i, j, trie, node, board, word, res):
        if not node or trie.num_of_words == 0:
            return
        if node.isWord:
            res.append(word)
            node.isWord = False
            trie.num_of_words -= 1

        tmp = board[i][j]
        board[i][j] = '#'
        if i + 1 < len(board):
            c = board[i + 1][j]
            self.dfs(i + 1, j, trie, node.children.get(c), board, word + c, res)
        if j + 1 < len(board[0]):
            c = board[i][j + 1]
            self.dfs(i, j + 1, trie, node.children.get(c), board, word + c, res)
        if i - 1 >= 0:
            c = board[i - 1][j]
            self.dfs(i - 1, j, trie, node.children.get(c), board, word + c, res)
        if j - 1 >= 0:
            c = board[i][j - 1]
            self.dfs(i, j - 1, trie, node.children.get(c), board, word + c, res)
        board[i][j] = tmp

    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        res = []
        trie = Trie()
        node = trie.root
        for w in words:
            trie.insert(w)
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(i, j, trie, node.children[board[i][j]], board, board[i][j], res)
        return res

    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        def same(p, q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            return p.val == q.val and same(p.left, q.left) and same(p.right, q.right)        
        return same(p, q)

    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stack = []
        node = root
        while node or stack:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            res.append(node.val)
            node = node.right
        return res

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def make(start, end):
            if start >= end:
                return None
            return TreeNode(
                val = nums[(start+end)//2],
                left = make(start, (start+end)//2),
                right = make((start+end)//2+1, end)
            )
        return make(0, len(nums))

    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return abs(self.getHeight(root.left) - self.getHeight(root.right)) < 2 and self.isBalanced(root.left) and self.isBalanced(root.right)
    def getHeight(self, node):
        if not node:
            return 0
        return max(self.getHeight(node.left), self.getHeight(node.right))+1

    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if not root.left and not root.right:
            return 1
        if root.left and not root.right:
            return 1 + self.minDepth(root.left)
        if root.right and not root.left:
            return 1 + self.minDepth(root.right)
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))

    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        if not root:
            return False
        if not root.left and not root.right and root.val == targetSum:
            return True
        targetSum -= root.val
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
    
    def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        res = []
        def dfs(root, sum, tmp):
            if root:
                if not root.left and not root.right and root.val == sum:
                    tmp.append(root.val)
                    return res.append(tmp)
                dfs(root.left, sum-root.val, tmp+[root.val])
                dfs(root.right, sum-root.val, tmp+[root.val])
        dfs(root, targetSum, [])
        return res 

    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.right)
                stack.append(node.left)
        return res

    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node:
                res.append(node.val)
                stack.append(node.left)
                stack.append(node.right)
        return res[::-1]
    
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        res = []
        if not root:
            return []
        def dfs(node, tmp):
            if node:
                if not node.left and not node.right:    
                    return res.append(tmp+str(node.val))
                if node.left:
                    dfs(node.left, tmp+str(node.val)+"->")
                if node.right:
                    dfs(node.right, tmp+str(node.val)+"->")
        dfs(root, "")
        return res

    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        if root.left and not root.left.left and not root.left.right:
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        if not root:
            return 0
        if not root.children:
            return 1
        heights = []
        for child in root.children:
            heights.append(self.maxDepth(child))
        return max(heights)+1


    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        def dfs(root):
            if not root:
                return 0
            l = dfs(root.left)
            r = dfs(root.right)
            self.ans += abs(l-r)
            return root.val + l+ r
        dfs(root)
        return self.ans
    
    def isSubtree(self, root, subRoot):
        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """
        if not root:
            return False
        if self.isSame(root, subRoot):
            return True
        return self.isSubtree(root.right, subRoot) or self.isSubtree(root.left, subRoot)
        
    def isSame(self, p, q):
        if not p and not q:
            return True
        if p and q:
            return p.val==q.val and self.isSame(p.right, q.right) and self.isSame(p.left, q.left)
        
    
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        res = []
        def dfs(node):
            if not node:
                return
            res.append(node.val)
            for child in node.children:
                dfs(child)
        dfs(root)
        return res

    def mergeTrees(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: TreeNode
        """
        if root1 and root2:
            root = TreeNode(root1.val+root2.val)
            root.left = self.mergeTrees(root1.left, root2.left)
            root.right = self.mergeTrees(root1.right, root2.right)
            return root
        else:
            return root1 or root2

    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        q = deque()
        q.append(root)
        res = []
        while q:
            l = len(q)
            temp = 0
            for i in range(len(q)):
                node = q.popleft()
                temp += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(temp/l)
        return res

    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        d = set()
        def find(node):
            if not node:
                return False
            complement = k - node.val
            if complement in d:
                return True
            d.add(node.val)
            return find(node.left) or find(node.right)  
        print(d)
        return find(root)

    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        def dfs(node):
            if not node:
                return node
            if node.val == val:
                return node
            return dfs(node.left) or dfs(node.right)
        return dfs(root)

    def leafSimilar(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: bool
        """
        leaf1 = []
        leaf2 = []
        def dfs(node, lst):
            if node:
                dfs(node.left, lst)
                if not node.left and not node.right:
                    lst.append(node.val)
                dfs(node.right, lst)
        dfs(root1, leaf1)
        dfs(root2, leaf2)
        return leaf1 == leaf2

    def increasingBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        stack = []
        node = root
        arr = []
        while stack or node:
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            arr.append(node.val)
            node = nosde.right
        
        root = curr = TreeNode(arr[0])
        for i in arr[1:]:
            curr.right = TreeNode(i)
            curr = curr.right
        return root
    
        # stack = []
        # node = root
        # arr = []
        # root = curr = TreeNode(-10)
        # while stack or node:
        #     while node:
        #         stack.append(node)
        #         node = node.left
        #     node = stack.pop()
        #     if curr.val == -10:
        #         curr.val = node.val
        #     else:
        #         curr.right = TreeNode(node.val)
        #         curr = curr.right
        #     node = node.right
        # return root


    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        self.res = 0
        def dfs(node):
            if not node:
                return
            if low <= node.val <= high:
                self.res += node.val
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return self.res

    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        res = []
        q = deque()
        q.append((root, None, 0))
        while q:
            if len(res) == 2:
                break
            node, parent, depth = q.popleft()
            if node.val == x or node.val == y:
                res.append((parent, depth))
            if node.left:
                q.append((node.left, node, depth+1))
            if node.right:
                q.append((node.right, node, depth+1))
        
        xP, xd = res[0]
        yP, yd = res[1]
        return xP!=yP and xd == yd

    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(node, res):
            if not node:
                return 0
            res = (res*2) + node.val
            if not node.left and not node.right:
                return res
            return dfs(node.left, res) + dfs(node.right, res)
        return dfs(root, 0)
    
    def lexicalOrder(self, n):
        """
        :type n: int
        :rtype: Lisst[int]
        """
        res = []
        curr = 1
        for i in range(n):
            res.append(curr)
            if curr * 10 <= n:
                curr *= 10
            else:
                if curr >= n:
                    curr //= 10
                curr += 1
                while curr % 10 == 0:
                    curr //= 10
        return res
    
        def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        q = deque()
        q.append((beginWord, 1))
        wordList = set(wordList)
        visit = set()
        if endWord not in wordList:
            return 0

        while q:
            word, count = q.popleft()
            if word == endWord:
                return count
            for i in range(len(word)):
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    new = word[:i] + j + word[i+1:]
                    if new not in visit and new in wordList:
                        q.append((new, count+1))
                        visit.add(new)
        return 0

    def divisorGame(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n % 2 == 0

    def countBits(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = [0]
        for i in range(1, n+1):
            res.append(res[i >> 1] + i%2)
        return res

    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i = 0
        j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        return i == len(s)

    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        if not cost:
            return 0
        dp = [0] * len(cost)
        dp[0] = cost[0]
        for i in range(1, len(cost)):
            dp[i] = cost[i] + min(dp[i-1], dp[i-2])
        return min(dp[-1], dp[-2])

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr = nums[0]
        res = nums[0]
        for num in nums[1:]:
            curr = max(num, curr+num)
            res = max(res, curr)
        return res

    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0] * n for i in range(m)]
        for i in range(0, n):
            dp[0][i] = 1
        for i in range(1, m):
            dp[i][0] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if obstacleGrid[-1][-1] == 1 or obstacleGrid[0][0] == 1:  
            return 0
        m = len(obstacleGrid[0])
        n = len(obstacleGrid)
        dp = [ [0]*m for i in range(n) ]
        dp[0][0] = 1
        for i in range(1, m):
            if obstacleGrid[0][i] != 1 and dp[0][i-1] != 0:
                dp[0][i] = 1
            else:
                dp[0][i] = 0
        for i in range(1, n):
            if obstacleGrid[i][0] != 1 and dp[i-1][0] != 0:
                dp[i][0] = 1
            else:
                dp[i][0] = 0
        for i in range(1, n):
            for j in range(1, m):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i][j-1] + dp[i-1][j]
        return dp[-1][-1]

    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m = len(grid[0])
        n = len(grid)
        dp = [ [0]*m for i in range(n) ]
        dp[0][0] = grid[0][0]
        for i in range(1, m):
            dp[0][i] = grid[0][i]+ dp[0][i-1]
        for j in range(1, n):
            dp[j][0] = grid[j][0]+ dp[j-1][0]
        for i in range(1, n):
            for j in range(1, m):
                dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or s[0]== "0":
            return 0
        dp = [0] * (len(s)+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, len(s)+1):
            if 0 < int(s[i-1:i]) <= 9:
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        return dp[-1]

    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        #dp[2] = dp[0] * dp[1] + dp[1] * dp[0]
        #dp[3] = dp[0] * dp[2] + dp[1] * dp[1] + dp[2] * dp[0]
        dp = [0] * (n+1)
        dp[0] = 1
        for i in range(1, n+1):
            for j in range(i):
                dp[i] += dp[j] * dp[i-1-j]
        return dp[-1]

    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        for i in range(len(triangle)-2, -1, -1):
            for j in range(i+1):
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
        return triangle[0][0]

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [False] * (len(s)+1)
        dp[0] = True
        for i in range(len(s)):
            for j in range(i, len(s)):
                if dp[i]:
                    if s[i:j+1] in wordDict:
                        dp[j+1] = True
        return dp[-1]

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        wordDict = set(wordDict)
        def helper(idx):
            if idx == len(s):
                return [""]
            res = []
            for j in range(idx+1, len(s)+1):
                if s[idx:j] in wordDict:
                    for tail in helper(j):
                        if tail != "":
                            res.append(s[idx:j] + " " + tail)
                        else:
                            res.append(s[idx:j])
            return res
        return helper(0)

    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        _max = nums[0]
        _min = nums[0]
        res = nums[0]
        for num in nums[1:]:
            _max *= num 
            _min *= num 
            _max, _min = max(_max, num, _min), min(_max, num, _min)
            res = max(res, _max)
        return res

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = [0, 0, 0] + nums
        for i in range(3, len(nums)):
            b2 = nums[i] + nums[i-3]
            b3 = nums[i] + nums[i-2]
            nums[i] = max(b2, b3)
        return max(nums)

    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums[0], nums[1])
        nums1 = [0,0,0] + nums[1:]
        nums2 = [0,0,0] + nums[:-1]
        for i in range(3, len(nums1)):
            b2 = nums1[i] + nums1[i-3]
            b3 = nums1[i] + nums1[i-2]
            nums1[i] = max(b2, b3)
        for i in range(3, len(nums2)):
            b2 = nums2[i] + nums2[i-3]
            b3 = nums2[i] + nums2[i-2]
            nums2[i] = max(b2, b3)
        return max( max(nums1), max(nums2) )

    
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        cool_down, sell, hold = 0, 0, -float('inf')
        
        for stock_price_of_Day_i in prices:
            prev_cool_down, prev_sell, prev_hold = cool_down, sell, hold
            hold = max(prev_hold, prev_cool_down - stock_price_of_Day_i)
            sell = prev_hold + stock_price_of_Day_i
            cool_down = max(prev_cool_down, prev_sell)
        return max(sell, cool_down)

    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        dp = [1] * (n+1)
        dp[0] = 0
        for num in range(2, n+1):
            i = 1
            j = num-1
            temp = 0
            while i <= j:
                temp = max(temp, (max(i, dp[i]) * max(j, dp[j])) )  
                i += 1
                j -= 1
            dp[num] = temp
        return dp[n] 

    def countNumbersWithUniqueDigits(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 1
        res = 10
        start = 9
        for i in range(1, n):
            start *= (10-i)
            res += start
        return res 

    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """        
        buy = float("-inf")
        sell = 0
        
        for today_price in prices:
            prev_buy = buy
            prev_sell = sell
            buy = max(prev_buy, prev_sell - today_price)
            sell = max(prev_sell, prev_buy + today_price - fee)
            print(buy, sell)
        return sell

    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        zero = 0
        one = 1
        if n == 0:
            return 0
        if n == 1:
            return 1
        res = 0
        for i in range(2, n+1):
            res = zero + one
            zero = one
            one = res
        return res

    def getMaximumGenerated(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        dp = [0] * (n+1)
        dp[0] = 0
        dp[1] = 1
        
        for i in range(2, n+1):
            if i % 2 == 0:
                dp[i] = dp[i//2]
            else:
                j = i //2
                dp[i] = dp[j] + dp[j+1]
        return max(dp)

    def sumSubarrayMins(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        sums = [0] * len(arr)
        stack = []
        for i in range(len(arr)):
            while stack and arr[stack[-1]] > arr[i]:
                stack.pop()
            if stack:
                j = stack[-1]
                sums[i] = sums[j] + arr[i] * (i-j)
            else:
                sums[i] = arr[i] * (i+1)
            stack.append(i)
        return sum(sums) % (10**9+7)

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        heights.append(0)
        res = 0
        stack = [-1]
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i -stack[-1] -1
                res = max(res, h*w)
            stack.append(i)
        return res

    def countOfAtoms(self, formula):
        """
        :type formula: str
        :rtype: str
        """
        dict = {}
        m = [1]
        digit = '' 
        lower = '' 
        for i in range(len(formula)-1, -1, -1):
            element = formula[i] + lower
            if element.isdigit():
                digit = element + digit       
            elif element.islower():
                lower = element      
            elif element == ')':
                m.append(m[-1] * int(digit or 1))
                digit = ''      
            elif element == '(':
                m.pop()  
            else:
                dict[element] = dict.get(element, 0) + m[-1]*int(digit or 1)
                digit = ''
                lower = ''
        output = ''
        for key, value in sorted(dict.items()):
            if value == 1:
                value = ''
            output = output + key + str(value)
        return output

    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for i in range(31, -1, -1):
            pool = set([num >> i for num in nums])
            res <<= 1
            #largest_possible in this length(same as res)
            largest_possible = res+1
            for a in pool:
                # a, b both in pool, largest = a ^ b => largest^a = a ^ b^a = b 
                # largest_possible ^ a = b
                if largest_possible ^ a in pool:
                    res = largest_possible
                    break
        return res

    def containsNearbyAlmostDuplicate(self, nums, indexDiff, valueDiff):
        """
        :type nums: List[int]
        :type indexDiff: int
        :type valueDiff: int
        :rtype: bool
        """
        buckets = {}
        valueDiff += 1
        for idx, num in enumerate(nums):
            bucketID = num // valueDiff
            if bucketID in buckets:
                return True
            for  i in (bucketID-1, bucketID+1):
                if i in buckets:
                    if abs(buckets[i] - num) < valueDiff:
                        return True
            buckets[bucketID] = num
            if idx >= indexDiff:
                del buckets[ nums[idx-indexDiff] // valueDiff]
        return False

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        cache = {}
        def dfs(i, j):
            if (i, j) in cache:
                return cache[(i, j)]
            if i >= len(s) and j >= len(p):
                return True
            if j >= len(p):
                return False
            match = i < len(s) and (s[i] == p[j] or p[j] == ".")
            if (j+1) < len(p) and p[j+1] == "*":
                cache[(i, j)] = dfs(i, j+2) or (match and dfs(i+1, j))
                return cache[(i, j)]
            if match:
                cache[(i, j)] = dfs(i+1, j+1)
                return cache[(i, j)]
            cache[(i, j)] = False
            return False
        return dfs(0, 0)

    def isAdditiveNumber(self, num):
        """
        :type num: str
        :rtype: bool
        """
        for i in range(1, len(num)):
            for j in range(i+1, len(num)):
                if num[0] == "0" and i > 1:
                    break
                if num[i] == "0" and j > i+1:
                    break
                num1 = int(num[:i])
                num2 = int(num[i:j])
                k = j
                while k < len(num):
                    num3 = num1 + num2
                    if num[k:].startswith(str(num3)):
                        k += len(str(num3))
                        num1 = num2
                        num2 = num3
                    else:
                        break
                if k == len(num):
                    return True
        return False

    def splitIntoFibonacci(self, num):
        """
        :type num: str
        :rtype: List[int]
        """
        ans = []
        def backtrack(ans, idx):
            if idx == len(num) and len(ans) >= 3:
                return True
            for i in range(idx, len(num)):
                if i > idx and num[idx] == "0":
                    break
                temp = int(num[idx:i+1])
                sz = len(ans)
                if temp >= 2**31:
                    break
                elif sz >= 2 and temp > ans[-1]+ans[-2]:
                    return False
                elif sz <= 1 or (sz >= 2 and temp == ans[-1]+ans[-2]):
                    ans.append(temp)
                    if backtrack(ans, i+1):
                        return True
                    ans.pop()
            return False
        backtrack(ans, 0)
        return ans

    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = [0]
        for i in range(n):
            for j in range(len(res)-1, -1, -1):
                res.append( res[j] | 1 << i )
        return res

    def predictTheWinner(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        def maxScore(i, j):
            if i > j:
                return 0
            a = nums[i] + min(maxScore(i+2, j), maxScore(i+1, j-1))
            b = nums[j] + min(maxScore(i, j-2), maxScore(i+1, j-1))
            return max(a, b)
        p1 = maxScore(0, len(nums)-1)
        return p1 >= (sum(nums)-p1)

    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        d = {0:1}
        res = 0
        prev = 0
        for num in nums:
            prev += num
            if prev - k in d:
                res += d[prev - k]
            if prev not in d:
                d[prev] = 1
            else:
                d[prev] += 1
        return res
    
    def isReflected(self, points):
        """
        :type points: List[List[int]]
        :rtype: bool
        """
        points_set = set(map(tuple, points))
        points.sort()
        min_x = points[0][0]
        max_x = points[-1][0] 
        mid_x = (max_x + min_x)/2.0
        for x, y in points_set:
            if (2*mid_x - x, y) not in points_set:
                return False               
        return True

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        edit = 0
        for num in nums:
            if num == val:
                continue
            else:
                nums[edit] = num
                edit += 1
        return edit

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        for i in range(len(haystack) - len(needle)+1):
            for j in range(len(needle)):
                if haystack[i+j] != needle[j]:
                    break
                if j == len(needle)-1:
                    return i
        return -1
    
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        s = s.strip()
        count = 0
        for i in range(len(s)-1, -1, -1):
            if s[i] != " ":
                count += 1
            else:
                break
        return count

    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        d = {}
        words = s.split()
        if len(pattern) != len(words) or len(set(pattern)) != len(set(words)):
            return False
        for idx, char in enumerate(pattern):
            if char not in d:
                d[char] = words[idx]
            else:
                if d[char] != words[idx]:
                    return False
        return True

    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s_d = {}
        for word in s:
            if word not in s_d:
                s_d[word] = 1
            else:
                s_d[word] += 1
        for word in t:
            if word not in s_d:
                return False
            else:
                s_d[word] -= 1
                if s_d[word] == 0:
                    del s_d[word]
        return len(s_d) == 0
    
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        d = {}
        for i in range(len(nums)):
            if nums[i] not in d:
                d[nums[i]] = i
            else:
                if abs(i - d[nums[i]]) <= k:
                    return True
                else:
                    d[nums[i]] = i
        return False

    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n:
            count += 1
            n = n & (n-1)
        return count

    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        if not nums:
            return []
        start = nums[0]
        count = 0
        res = []
        for i in range(1, len(nums)):
            if nums[i] != start+count+1:
                if count == 0:
                    res.append(str(start))
                else:
                    res.append(str(start)+"->"+str((start+count)))
                start = nums[i]
                count = 0
            else:
                count += 1
        if start != nums[-1]:
            res.append(str(start)+"->"+str((start+count)))
        else:
            res.append(str(nums[-1]))
        return res

    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not k or not head:
            return head
        length = 1
        last = head
        while last.next:
            length += 1
            last = last.next
        last.next = head

        for i in range(length - (k%length)):
            last = last.next
        dummy = last.next
        last.next = None
        return dummy

    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        if head.val != head.next.val:
            head.next = self.deleteDuplicates(head.next)
            return head
        if not head.next.next or head.next.val != head.next.next.val:
            return self.deleteDuplicates(head.next.next)
        return self.deleteDuplicates(head.next)

    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if not head:
            return None
        d = {}
        curr = head
        while curr:
            d[curr] = Node(curr.val, None, None)
            curr = curr.next
        curr = head
        while curr:
            if curr.next:
                d[curr].next = d[curr.next]
            if curr.random:
                d[curr].random = d[curr.random]
            curr = curr.next
        return d[head]

    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        res = ""
        storeIntRoman = [[1000, "M"], [900, "CM"], [500, "D"], [400, "CD"], [100, "C"], [90, "XC"], [50, "L"], [40, "XL"], [10, "X"], [9, "IX"], [5, "V"], [4, "IV"], [1, "I"]]
        for i in range(len(storeIntRoman)):
            while num >= storeIntRoman[i][0]:
                res += storeIntRoman[i][1]
                num -= storeIntRoman[i][0]
        return res

    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [1] * len(nums)
        for i in range(1, len(nums)):
            res[i] = res[i-1]*nums[i-1]
        temp = 1
        for i in range(len(nums)-2, -1, -1):
            temp *= nums[i+1]
            res[i] *= temp
        return res

    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        citations.sort(reverse=True)
        for idx, citation in enumerate(citations):
            if idx >= citation:
                return idx
        return len(citations)

    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [0] * len(nums)
        l = 0
        r = len(nums)-1
        for i in range(len(nums)-1, -1, -1):
            if abs(nums[l]) < abs(nums[r]):
                res[i] = nums[r] ** 2
                r -= 1
            else:
                res[i] = nums[l] ** 2
                l += 1
        return res

    def smallerNumbersThanCurrent(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        sort_nums = sorted(nums)
        d = dict()
        for idx, num in enumerate(sort_nums):
            if num not in d:
                d[num] = idx
        res = []
        for num in nums:
            res.append(d[num])
        return res

    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        edit = 0
        for num in nums:
            if num != 0:
                nums[edit] = num
                edit += 1
        while edit < len(nums):
            nums[edit] = 0
            edit += 1

    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        leftSum = 0
        rightSum = sum(nums) - nums[0]
        idx = 0
        while idx < len(nums):
            if leftSum == rightSum:
                return idx
            leftSum += nums[idx]
            idx += 1
            if idx < len(nums):
                rightSum -= nums[idx]
        return -1

    def sortArrayByParityII(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        even = 0
        odd = 1
        while even < len(nums) or odd < len(nums):
            while even < len(nums) and nums[even] % 2 == 0:
                even += 2
            while odd < len(nums) and nums[odd] % 2 != 0:
                odd += 2
            if even < len(nums) and odd < len(nums):
                nums[odd], nums[even] = nums[even], nums[odd]
            even += 2
            odd += 2
        return nums

    def validMountainArray(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        p1 = 0
        p2 = len(arr)-1
        while p1+1 < len(arr) and arr[p1] < arr[p1+1]:
            p1 += 1
        while p2 > 0 and arr[p2] < arr[p2-1]:
            p2 -= 1
        return 0 < p1==p2 < len(arr)-1
    
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        def reverse(start, end):
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
        k = k % len(nums)
        reverse(0, len(nums)-k-1)
        reverse(len(nums)-k, len(nums)-1)
        reverse(0, len(nums)-1)

    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        matrix = [[None]*n for i in range(n)]
        top = 0
        left = 0
        right = n-1
        bottom = n-1
        numbers = (i for i in range(1, n**2+1))
        while left <= right and top <= bottom:
            for i in range(left, right+1):
                matrix[top][i] = next(numbers)
            top += 1
            for i in range(top, bottom+1):
                matrix[i][right] = next(numbers)
            right -= 1
            for i in range(right, left-1, -1):
                matrix[bottom][i] = next(numbers)
            bottom -= 1
            for i in range(bottom, top-1, -1):
                matrix[i][left] = next(numbers)
            left+= 1
        return matrix

    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        rows = [set() for i in range(9)]
        cols = [set() for i in range(9)]
        boxes = [set() for i in range(9)]
        for i in range(len(board)):
            for j in range(len(board[0])):
                box = (i//3)*3 + j//3
                if board[i][j] == ".":
                    continue
                if board[i][j] in rows[i]:
                    return False
                if board[i][j] in cols[j]:
                    return False
                if board[i][j] in boxes[box]:
                    return False
                rows[i].add(board[i][j])
                cols[j].add(board[i][j])
                boxes[box].add(board[i][j])
        return True

    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        live = set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                curr = board[i][j]
                count_1 = 0
                if j-1 >= 0:
                    if i-1 >= 0 and board[i-1][j-1] == 1:
                        count_1 += 1
                    if board[i][j-1] == 1:
                        count_1 += 1
                    if i+1 < len(board) and board[i+1][j-1] == 1:
                        count_1 += 1
                if j+1 < len(board[0]):
                    if i-1 >= 0 and board[i-1][j+1] == 1:
                        count_1 += 1
                    if board[i][j+1] == 1:
                        count_1 += 1
                    if i+1 < len(board) and board[i+1][j+1] == 1:
                        count_1 += 1
                if i-1 >= 0 and board[i-1][j] == 1:
                    count_1 += 1
                if i+1 < len(board) and board[i+1][j] == 1:
                    count_1 += 1
                if curr == 1 and (count_1 == 2 or count_1 == 3):
                    live.add((i, j))
                if curr == 0 and count_1 == 3:
                    live.add((i, j))

        for i in range(len(board)):
            for j in range(len(board[0])):
                if (i, j) in live:
                    board[i][j] = 1
                else:
                    board[i][j] = 0

    def maxSubarraySumCircular(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        total = 0
        maxCur = 0
        maxSum = nums[0]
        minCur = 0
        minSum = nums[0]
        for num in nums:
            total += num
            maxCur = max(num, maxCur + num)
            maxSum = max(maxSum, maxCur)
            minCur = min(num, minCur + num)
            minSum = min(minSum, minCur)
        return max(maxSum, total-minSum) if maxSum > 0 else maxSum


    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        matrix.reverse()
        for i in range(len(matrix)):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        l = 0
        r = len(s)-1
        while l < r:
            s[l], s[r] = s[r], s[l]
            l+=1
            r-=1

    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        s = list(s) 
        start = 0
        end = 0
        count = 0
        while end < len(s):
            if len(s[start:]) < k:
                break
            for i in range(k-1):
                end += 1
            if count%2 == 0 and end < len(s):
                end2 = end
                while start < end2:
                    s[start], s[end2] = s[end2], s[start]
                    start += 1
                    end2 -= 1   
            count += 1
            end += 1
            start = end  
        if count%2 == 0 and start < len(s):
            end = len(s)-1
            while start < end:
                s[start], s[end] = s[end], s[start]
                start += 1
                end -= 1 
        return "".join(s)

    def backspaceCompare(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s_stack = []
        t_start = []
        for i in range(len(s)):
            if s_stack and s[i] == "#":
                s_stack.pop()
            elif not s_stack and s[i] == "#":
                continue
            else:
                s_stack.append(s[i])
        for i in range(len(t)):
            if t_start and t[i] == "#":
                t_start.pop()
            elif not t_start and t[i] == "#":
                continue
            else:
                t_start.append(t[i])
        return s_stack == t_start
    
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        rep = ""
        for i in range(len(s)//2):
            rep += s[i]
            if len(s) % len(rep) == 0:
                if rep * (len(s) // len(rep)) == s:
                    return True
        return False